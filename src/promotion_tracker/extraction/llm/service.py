"""Integração com o Ollama para classificar e extrair ofertas de imagens.

Este módulo possui o prompt, prepara imagens para inferência, executa as
consultas principal e de revisita e gerencia o processo local do Ollama. O
orquestrador continua responsável pelo fluxo das imagens; ``OllamaService``
fica responsável pela fronteira com o modelo e pelo ciclo de vida do servidor.

Ajustes para máquinas com pouca memória (ex.: 16 GB de RAM e GPU de 8 GB de
VRAM): ao iniciar o servidor por este programa, são definidas variáveis de
ambiente que reduzem o uso de VRAM (ver ``SERVER_ENV``). Se já houver um
servidor Ollama rodando, essas variáveis NÃO têm efeito; reinicie o servidor.

Encartes muito densos: a geração em JSON pode entrar em loop ou estourar o
limite de tokens (cada falha custava ~90 s e era repetida). Agora:
  1. a geração é monitorada e interrompida ao detectar repetição ou ao passar
     de ``MAX_CALL_SECONDS``;
  2. as ofertas já completas são aproveitadas da resposta interrompida;
  3. só nesse caso o encarte é relido em faixas menores com sobreposição, e
     os resultados são unidos sem duplicatas. Faixas que ainda falham são
     subdivididas automaticamente.
Encartes pequenos e imagens que não são encarte terminam normalmente na
primeira chamada e não passam por nada disso.

Resultados suspeitos: se a resposta "completa" agrupa produtos demais num único
preço (``MAX_SHARED_VARIANTS``), repete o mesmo produto em ofertas diferentes,
ou retorna muito menos ofertas do que o esperado para o tamanho da imagem, o
modelo provavelmente juntou quadros do encarte ou perdeu linhas. Nesse caso
as ofertas gigantes são descartadas e o encarte é relido em faixas.
"""

from __future__ import annotations

import io
import json
import logging
import math
import os
import re
import subprocess
import time

import ollama
from PIL import Image, ImageDraw
from pydantic import BaseModel, Field, ValidationError

from .. import settings as cfg
from ..ocr.matching import normalize_search_text
from ..settings import (
    COMPOSITE_GAP,
    COMPOSITE_MAX_HEIGHT,
    DENSE_NUM_PREDICT,
    MAX_LADO,
    MIN_LADO_ENVIO,
    NUM_BATCH,
    NUM_CTX,
    NUM_PREDICT,
    NUM_THREAD,
    OLLAMA_START_TIMEOUT,
)
from ..silver_writer import SilverWriter, has_repeated_products
from .schemas import (
    CURRENT_YEAR,
    RespostaCortada,
    ResultadoExtracao,
)

logger = logging.getLogger(__name__)

# Estas duas constantes eram usadas no modo denso sem serem importadas
# (NameError). Se existirem em settings.py, são usadas; senão há um padrão
# seguro: mesmo contexto do modo normal e GPU automática (-1).
DENSE_NUM_CTX: int = getattr(cfg, "DENSE_NUM_CTX", NUM_CTX)
DENSE_NUM_GPU: int = getattr(cfg, "DENSE_NUM_GPU", -1)

# Quanto tempo o modelo fica carregado após cada chamada.
KEEP_ALIVE = "5m"

# --- Proteção contra respostas longas demais / em loop ----------------------
# AJUSTE: tempo máximo de geração maior no modo normal (antes 60 s), para
# não cortar encartes médios (15–25 ofertas) no meio da geração.
MAX_CALL_SECONDS: float = float(getattr(cfg, "MAX_CALL_SECONDS", 90))
# Confere repetição a cada N fragmentos recebidos.
LOOP_CHECK_EVERY = 24

# AJUSTE: faixas menores e mais numerosas. Faixas grandes demais estouravam
# BAND_NUM_PREDICT e perdiam ofertas no fim; com faixas menores cada chamada
# cabe no orçamento de tokens e tempo.
BAND_RATIO: float = float(getattr(cfg, "BAND_RATIO", 0.40))
BAND_OVERLAP: float = float(getattr(cfg, "BAND_OVERLAP", 0.15))
MIN_BAND_PX: int = int(getattr(cfg, "MIN_BAND_PX", 160))
MAX_BANDS: int = int(getattr(cfg, "MAX_BANDS", 12))
BAND_NUM_PREDICT: int = int(getattr(cfg, "BAND_NUM_PREDICT", 1024))
BAND_MAX_SECONDS: float = float(getattr(cfg, "BAND_MAX_SECONDS", 25))
# AJUSTE: profundidade máxima de subdivisão de uma faixa que falhou.
BAND_MAX_DEPTH: int = int(getattr(cfg, "BAND_MAX_DEPTH", 2))

# Uma oferta "compartilhada" legítima (ex.: "iogurte morango OU coco") tem
# poucas variantes. Acima disso, o modelo provavelmente agrupou quadros
# diferentes do encarte numa oferta só.
MAX_SHARED_VARIANTS: int = int(getattr(cfg, "MAX_SHARED_VARIANTS", 8))

# AJUSTE: cobertura mínima esperada (ofertas por megapixel). Abaixo disso,
# consideramos que a resposta "completa" perdeu linhas e disparamos releitura.
# Valor conservador; ajuste empiricamente conforme seus encartes.
MIN_OFFERS_PER_MP: float = float(getattr(cfg, "MIN_OFFERS_PER_MP", 2.0))

_RE_ENCARTE_VALIDO = re.compile(r'"encarte_valido"\s*:\s*(true|false)')
_RE_VALIDADE = re.compile(r'"validade"\s*:\s*(null|"[^"]*")')

# Variáveis de ambiente aplicadas ao ``ollama serve`` iniciado por este
# programa. Valores já definidos no ambiente do usuário têm prioridade.
SERVER_ENV: dict[str, str] = {
    "OLLAMA_FLASH_ATTENTION": "1",
    "OLLAMA_KV_CACHE_TYPE": "q8_0",
    "OLLAMA_NUM_PARALLEL": "1",
    "OLLAMA_MAX_LOADED_MODELS": "1",
}


def is_looping(text: str) -> bool:
    """Detecta saída degenerada: só espaços ou um trecho repetido no final."""
    if len(text) < 400:
        return False
    if not text[-200:].strip():
        return True
    for period in range(8, 301):
        repeats = 8 if period < 40 else 4
        span = period * repeats
        if len(text) < span:
            break
        if text[-span:] == text[-period:] * repeats:
            return True
    return False


def salvage_offers(text: str) -> dict | None:
    """Recupera as ofertas completas de uma resposta JSON interrompida."""
    key = text.find('"ofertas"')
    if key == -1:
        return None
    start = text.find("[", key)
    if start == -1:
        return None
    decoder = json.JSONDecoder()
    position = start + 1
    offers: list[dict] = []
    seen: set[str] = set()
    while True:
        while position < len(text) and text[position] in " \t\r\n,":
            position += 1
        if position >= len(text) or text[position] != "{":
            break
        try:
            offer, position = decoder.raw_decode(text, position)
        except json.JSONDecodeError:
            break
        signature = json.dumps(offer, sort_keys=True, ensure_ascii=False)
        if signature not in seen:
            seen.add(signature)
            offers.append(offer)
    if not offers:
        return None
    head = text[:start]
    valid_match = _RE_ENCARTE_VALIDO.search(head) or _RE_ENCARTE_VALIDO.search(text)
    date_match = _RE_VALIDADE.search(head) or _RE_VALIDADE.search(text)
    return {
        "encarte_valido": valid_match.group(1) == "true" if valid_match else True,
        "validade": json.loads(date_match.group(1)) if date_match else None,
        "ofertas": offers,
    }


PROMPT = f"""
Você analisa imagens de encartes de supermercados brasileiros.

CLASSIFICAÇÃO (seja rigoroso):
- encarte_valido=true SOMENTE se a imagem mostrar, em formato de material
  promocional de supermercado, produtos com PREÇO em reais (R$ ou
  números grandes de preço) ao lado de cada produto. Um recorte de
  encarte também vale, desde que tenha produto E preço visíveis.
- encarte_valido=false para: foto de produto, embalagem ou prateleira sem
  etiqueta de preço; foto pessoal, paisagem, comida, receita; logo,
  banner institucional, anúncio de marca, cupom, print de rede social,
  meme, texto solto; imagem sem nenhum preço legível.
- Na dúvida, ou se não houver preço legível: encarte_valido=false.
- Se false: ofertas=[] e validade=null. Não invente.

RECORTES:
- A imagem pode ser um recorte do encarte, ou vários recortes empilhados
  na vertical (separados por uma linha cinza). Extraia só ofertas
  COMPLETAS (nome e preço visíveis); ignore itens cortados na borda.

SE ENCARTE, extraia TODAS as ofertas.

COMO SEPARAR AS OFERTAS (REGRA MAIS IMPORTANTE):
- Encartes são grades de QUADROS. Cada quadro tem o NOME do produto
  impresso e o SEU PRÓPRIO preço. Um quadro = UMA oferta, com o preço
  daquele quadro.
- NUNCA junte produtos de quadros diferentes na mesma oferta, mesmo que
  os preços pareçam iguais. Se o encarte tem 12 quadros, o resultado tem
  cerca de 12 ofertas (cada uma com o seu preço), e NÃO uma oferta com
  12 variantes.
- Uma oferta só tem MAIS DE UMA variante quando o TEXTO impresso do MESMO
  quadro lista vários produtos ligados por "OU", "/", vírgula ou "e", ou
  diz "sabores"/"variados". Decida pelo TEXTO do quadro, NUNCA pela foto:
  a foto pode mostrar duas embalagens do mesmo produto, e isso é UMA
  variante só.
- Cada produto aparece UMA única vez no resultado, em UMA única oferta.
  Nunca repita um produto com outro preço.
- Copie só o que está escrito no quadro; não invente produtos nem preços.

COBERTURA (OBRIGATÓRIO):
- Percorra a imagem INTEIRA, da esquerda para a direita e de cima para
  baixo. NÃO pare no meio da grade.
- Estime quantos quadros com preço existem na imagem e produza uma
  oferta para CADA um. Se a imagem tem 4 colunas × 5 linhas, o
  resultado deve ter ~20 ofertas.
- Se estiver em dúvida se um quadro tem preço legível, inclua a oferta
  mesmo assim (com o preço que conseguir ler); é melhor incluir do que
  omitir.

VALIDADE (campo "validade"):
- Procure na imagem por textos como "Válido até", "Validade:",
  "Ofertas válidas de X a Y", "Promoção válida até".
- Devolva SEMPRE no formato ISO "YYYY-MM-DD" (ex.: "2026-10-25").
- Se a imagem trouxer só dia/mês ("até 25/10") use o ano corrente
  ({CURRENT_YEAR}); se essa data já passou há mais de 30 dias, use
  {CURRENT_YEAR + 1}.
- Se a imagem trouxer mês por extenso ("25 de outubro de 2026"), converta
  para ISO.
- Se a imagem trouxer APENAS um dia da semana ("terça-feira") sem data,
  devolva null (não é data absoluta).
- Se não houver validade na imagem, devolva null.

VARIANTE (por produto):
- produto: nome completo como escrito no quadro, começando pelo tipo do
  produto (ex.: "Doce de Leite Maranguape Pote 320g").
- marca: identificável ou null.
- quantidade e unidade_medida: SEMPRE a medida escrita no nome.
  "390g"→390 "g"; "1,5L"→1.5 "l"; "2L"→2 "l"; "500ml"→500 "ml".
  Use "un" apenas quando o nome trouxer só contagem ("12 unidades") e
  nenhum peso/volume. Nunca assuma 1 "un" se houver peso ou volume.
- unidade_medida: "g"|"kg"|"mg"|"ml"|"l"|"un"|"outro"|null.

OFERTA (por preço):
- preco_atual: preço principal em reais (decimal). Sempre presente.
- preco_anterior: preço anterior explícito ou null.
- preco_condicao: SEGUNDO preço da oferta, quando aparece com uma
  CONDIÇÃO (Clube, App, cartão, "leve 3 pague 2", "só para membros").
  É o preço menor, impresso logo abaixo/ao lado do principal, com uma
  etiqueta colorida ou em negrito da condição. Ex. na tela:
      "R$ 25,90"
      "R$ 22,90 no Clube Mais"
  → preco_atual=25.90, preco_condicao=22.90, condicao="Clube Mais".
  Se houver SÓ um preço, preco_condicao=null e condicao=null.
- condicao: texto EXATO da etiqueta da condição, como está escrito
  ("Clube Mais", "App", "Cartão Fidelidade", "leve 3 pague 2"). SEMPRE
  preencha se houver preco_condicao. SEMPRE deixe null se não houver
  segundo preço com condição.

VÁRIOS PRODUTOS NO MESMO QUADRO (mesmo preço):
Produtos ligados por "OU", "/", vírgula ou "e" no texto de UM quadro, sob
um único preço → uma oferta com várias variantes. Ex.: "IOGURTE NESTLÉ
MORANGO OU COCO 170G R$2,49" → 2 variantes (quantidade=170,
unidade_medida="g"), preco_atual=2.49. "FEIJÃO OU ARROZ 1KG R$9,99" →
2 variantes. Vários sabores não nomeados → variante "Sabores".

EXEMPLOS:
1) "ARROZ TIO JOÃO 5KG — R$25,90 / R$22,90 Clube Mais"
   variante: {{produto:"Arroz Tio João 5kg", marca:"Tio João",
              quantidade:5, unidade_medida:"kg"}}
   oferta:   {{preco_atual:25.90, preco_condicao:22.90,
              condicao:"Clube Mais"}}

2) "CAFÉ PILÃO 500G — R$18,90"  (só um preço)
   variante: {{produto:"Café Pilão 500g", quantidade:500, unidade_medida:"g"}}
   oferta:   {{preco_atual:18.90, preco_anterior:null,
              preco_condicao:null, condicao:null}}

3) Dois quadros lado a lado: "FEIJÃO KICALDO 1KG R$6,99" e
   "MACARRÃO COSTOSO 400G R$2,79"
   → DUAS ofertas, cada uma com uma variante e o seu preço.
   NÃO é uma oferta com 2 variantes (são quadros diferentes).

REGRAS:
1. Não invente dados. Associe cada preço ao produto do MESMO quadro.
2. Vírgula é decimal: "R$12,99"→12.99; "1,5L"→1.5.
3. Não confunda peso/volume/código/parcelas com preço.
4. "REF:1234" é código, não preço. Ignore "CADA" junto ao preço.
5. Não descarte oferta por campo secundário ilegível.
6. Não duplique ofertas nem produtos.
7. `condicao` só existe junto de `preco_condicao`. Se um dos dois está
   presente, o outro TAMBÉM deve estar.
8. Liste cada oferta UMA única vez. Quando acabarem os produtos visíveis,
   feche a lista e termine; nunca repita ofertas para completar espaço.
9. NÃO pare no meio da grade. Percorra TODOS os quadros antes de fechar
   o JSON.
"""

PROMPT_PRODUCTS = """
Você recebe uma imagem de encarte de supermercado e uma LISTA DE PRODUTOS
DE REFERÊNCIA. Sua tarefa é confirmar quais itens da lista aparecem na
imagem COMO O PRODUTO QUE ESTÁ À VENDA.

LISTA DE REFERÊNCIA:
{lista}

COMO DECIDIR (leia o NOME IMPRESSO do produto, não a foto):
1. Para cada produto do encarte, identifique o TIPO PRINCIPAL dele: é o que
   o produto É. "Queijo Coalho Maranguape 400g" é um queijo.
   "Nugget Sadia Presunto e Queijo" é um empanado (nugget), não um queijo.
2. Um item da lista só vale se o produto à venda FOR daquele tipo.
3. NÃO vale quando a palavra da lista aparece apenas como sabor, recheio,
   ingrediente, acompanhamento ou parte do nome de OUTRO tipo de produto.
   Exemplos:
   - lista "Queijo Coalho": "Queijo Coalho Maranguape" → vale;
     "Nugget Presunto e Queijo", "Pão de Queijo", "Biscoito sabor queijo"
     → NÃO valem.
   - lista "Doce de Leite": "Doce de Leite Maranguape 320g" → vale;
     "Biscoito recheado Doce de Leite", "Sorvete sabor doce de leite"
     → NÃO valem.
   - lista "Manteiga": "Manteiga Maranguape 500g" → vale;
     "Biscoito amanteigado", "Pipoca sabor manteiga" → NÃO valem.
4. Variações de marca, tamanho, sabor, plural e abreviação do MESMO tipo
   de produto valem (ex.: lista "Requeijão" casa com "Requeijão Cremoso
   Tirolez 200g").
5. Só confirme o que você consegue LER na imagem. Se não vê o produto,
   não inclua. Nunca complete a lista "por precaução".
6. Para cada item confirmado devolva:
   - texto_na_imagem: o nome do produto exatamente como está impresso;
   - item_da_lista: o item da lista, copiado LITERALMENTE como está;
   - e_o_proprio_produto: true se o produto impresso É daquele tipo;
     false se a palavra aparece só como sabor/recheio/ingrediente.
7. Se nenhum item da lista aparece, devolva a lista vazia.
8. Não repita itens.
"""


class _ProdutoConfirmado(BaseModel):
    """Um item de referência confirmado na imagem, com a evidência lida."""
    texto_na_imagem: str
    item_da_lista: str
    e_o_proprio_produto: bool


class _RevisitaProdutos(BaseModel):
    """Resposta estruturada da revisita de produtos."""
    confirmados: list[_ProdutoConfirmado] = Field(default_factory=list)


class OllamaService:
    """Executa inferências de visão e gerencia o servidor Ollama."""

    def __init__(
        self,
        model_name: str,
        segment_products: list[str],
        debug: bool,
        silver_writer: SilverWriter,
    ) -> None:
        self.model_name = model_name
        self.segment_products = segment_products
        self.debug = debug
        self.silver_writer = silver_writer
        self.model_used = False
        self._process: subprocess.Popen | None = None

    def start_server_if_needed(self) -> None:
        """Inicia ``ollama serve`` se nenhum servidor responder."""
        try:
            ollama.list()
            logger.info(
                "Ollama já está disponível; usando o servidor existente "
                "(as variáveis de otimização de memória não foram aplicadas)."
            )
            return
        except Exception:
            pass

        env = os.environ.copy()
        for key, value in SERVER_ENV.items():
            env.setdefault(key, value)

        creation_flags = 0
        if os.name == "nt":
            creation_flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        try:
            self._process = subprocess.Popen(
                ["ollama", "serve"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creation_flags=creation_flags,
                env=env,
            )
        except FileNotFoundError:
            raise RuntimeError("Executável 'ollama' não encontrado.") from None

        deadline = time.monotonic() + OLLAMA_START_TIMEOUT
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                self._process = None
                raise RuntimeError(
                    "O processo 'ollama serve' encerrou durante a inicialização."
                )
            try:
                ollama.list()
                logger.info("Servidor Ollama iniciado por este programa.")
                return
            except Exception:
                time.sleep(0.4)
        self.stop_owned_server()
        raise TimeoutError("O servidor Ollama não ficou disponível.")

    def stop_owned_server(self) -> None:
        """Encerra somente o processo Ollama iniciado por esta instância."""
        process = self._process
        self._process = None
        if process is None or process.poll() is not None:
            return
        logger.info("Encerrando o servidor Ollama iniciado por este programa...")
        try:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    logger.error("O processo Ollama não encerrou após kill().")
        except Exception:
            logger.exception("Falha ao encerrar o servidor Ollama iniciado.")

    @staticmethod
    def stop_existing_servers() -> None:
        """Encerra processos Ollama preexistentes usando comando do sistema."""
        logger.warning(
            "Encerrando processos Ollama já em execução "
            "(ENCERRAR_SERVIDOR_EXISTENTE=True)..."
        )
        try:
            if os.name == "nt":
                for image_name in ("ollama app.exe", "ollama.exe"):
                    subprocess.run(
                        ["taskkill", "/F", "/IM", image_name],
                        capture_output=True, timeout=10,
                    )
            else:
                subprocess.run(
                    ["pkill", "-x", "ollama"], capture_output=True, timeout=10
                )
        except Exception:
            logger.exception("Falha ao encerrar servidor Ollama existente.")

    def close(self, terminate_existing: bool) -> None:
        """Descarrega o modelo ou encerra o processo conforme configuração."""
        model_used = self.model_used
        self.model_used = False
        will_terminate = self._process is not None or (
            model_used and terminate_existing
        )
        if model_used and not will_terminate:
            logger.info("Solicitando descarregamento do modelo %s...", self.model_name)
            try:
                ollama.generate(model=self.model_name, prompt="", keep_alive=0)
                logger.info("Solicitação de descarregamento enviada.")
            except Exception:
                logger.warning("Não foi possível solicitar o descarregamento.")
        if self._process is not None:
            self.stop_owned_server()
        elif model_used and terminate_existing:
            self.stop_existing_servers()

    @staticmethod
    def stack_crops(crops: list[Image.Image]) -> Image.Image:
        """Empilha recortes verticalmente, separados por linha cinza."""
        if len(crops) == 1:
            return crops[0]
        width = max(crop.width for crop in crops)
        height = sum(crop.height for crop in crops) + COMPOSITE_GAP * (len(crops) - 1)
        canvas = Image.new("RGB", (width, height), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        y_position = 0
        for index, crop in enumerate(crops):
            canvas.paste(crop, (0, y_position))
            y_position += crop.height
            if index < len(crops) - 1:
                middle = y_position + COMPOSITE_GAP // 2
                draw.line(
                    [(0, middle), (width, middle)],
                    fill=(150, 150, 150), width=2,
                )
                y_position += COMPOSITE_GAP
        return canvas

    def make_crops(
        self, image: Image.Image, regions: list
    ) -> list[Image.Image]:
        """Recorta regiões e divide lotes que excederiam a altura máxima."""
        crops = [image.crop(region.box) for region in regions]
        batches: list[list[Image.Image]] = []
        current: list[Image.Image] = []
        height = 0
        for crop in crops:
            new_height = height + crop.height + (COMPOSITE_GAP if current else 0)
            if current and new_height > COMPOSITE_MAX_HEIGHT:
                batches.append(current)
                current, new_height = [], crop.height
            current.append(crop)
            height = new_height
        if current:
            batches.append(current)
        return [self.stack_crops(batch) for batch in batches]

    @staticmethod
    def prepare_image(image: Image.Image) -> tuple[bytes, dict]:
        """Redimensiona uma imagem e serializa seus bytes PNG para o modelo."""
        image = image.convert("RGB")
        original_name = image.size
        maior = max(image.size)
        if maior > MAX_LADO:
            image.thumbnail((MAX_LADO, MAX_LADO), Image.Resampling.LANCZOS)
        elif maior < MIN_LADO_ENVIO:
            fator = MIN_LADO_ENVIO / maior
            image = image.resize(
                (round(image.width * fator), round(image.height * fator)),
                Image.Resampling.LANCZOS,
            )
        resized_size = image.size
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()
        info = {
            "original": original_name,
            "enviada": resized_size,
            "kb": len(image_bytes) / 1024,
            "tokens_estimados": (resized_size[0] // 28) * (resized_size[1] // 28),
        }
        return image_bytes, info

    @staticmethod
    def build_options(
        *,
        dense: bool = False,
        attempt: int = 1,
        num_predict: int | None = None,
    ) -> dict:
        """Monta as opções de inferência do Ollama."""
        options = {
            "temperature": 0,
            "num_ctx": DENSE_NUM_CTX if dense else NUM_CTX,
            "num_predict": (
                num_predict
                if num_predict is not None
                else (DENSE_NUM_PREDICT if dense else NUM_PREDICT)
            ),
            "num_thread": NUM_THREAD,
            "num_batch": NUM_BATCH,
            "num_gpu": DENSE_NUM_GPU if dense else -1,
        }
        if attempt > 1:
            options.update(
                {"temperature": 0.3, "repeat_penalty": 1.1, "seed": attempt}
            )
        return options

    def _call_model(
        self,
        image: Image.Image,
        image_path,
        number: int,
        total_count: int,
        *,
        attempt: int = 1,
        dense: bool = False,
        max_seconds: float = MAX_CALL_SECONDS,
        num_predict: int | None = None,
        debug_number: int | None = None,
    ) -> tuple[str, str]:
        """Faz uma chamada em streaming e a interrompe se sair do controle."""
        image_bytes, info = self.prepare_image(image)
        logger.info(
            "Imagem %d/%d: %dx%d -> %dx%d (%.0f KB, ≈%d tokens de visão)",
            number, total_count, *info["original"], *info["enviada"],
            info["kb"], info["tokens_estimados"],
        )
        debug_id = debug_number if debug_number is not None else number
        if self.debug:
            self.silver_writer.save_debug_image(image_path, image, debug_id)

        start_time = time.monotonic()
        first_token_time: float | None = None
        chunks: list[str] = []
        last_chunk = None
        reason = "stop"
        self.model_used = True
        options = self.build_options(
            dense=dense, attempt=attempt, num_predict=num_predict
        )
        stream = ollama.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": PROMPT, "images": [image_bytes]}],
            format=ResultadoExtracao.model_json_schema(),
            options=options,
            stream=True,
            keep_alive=KEEP_ALIVE,
        )
        try:
            for chunk in stream:
                now = time.monotonic()
                if first_token_time is None:
                    first_token_time = now
                    logger.info("Primeiro token após %.1fs.", now - start_time)
                chunks.append(chunk["message"]["content"])
                last_chunk = chunk
                if len(chunks) % LOOP_CHECK_EVERY == 0:
                    if is_looping("".join(chunks)):
                        reason = "loop"
                        break
                    if now - first_token_time > max_seconds:
                        reason = "timeout"
                        break
        finally:
            close_stream = getattr(stream, "close", None)
            if callable(close_stream):
                try:
                    close_stream()
                except Exception:
                    pass

        response = "".join(chunks)
        total_duration = time.monotonic() - start_time
        if last_chunk is not None:
            def get_value(key: str):
                return (
                    last_chunk.get(key) if isinstance(last_chunk, dict)
                    else getattr(last_chunk, key, None)
                )

            prompt_count = get_value("prompt_eval_count")
            prompt_duration = get_value("prompt_eval_duration")
            eval_count = get_value("eval_count")
            eval_duration = get_value("eval_duration")
            if prompt_count and prompt_duration:
                logger.info(
                    "Prefill: %d tokens em %.1fs (%.1f tok/s)",
                    prompt_count, prompt_duration / 1e9,
                    prompt_count / (prompt_duration / 1e9),
                )
            if eval_count and eval_duration:
                logger.info(
                    "Geração: %d tokens em %.1fs (%.1f tok/s)",
                    eval_count, eval_duration / 1e9,
                    eval_count / (eval_duration / 1e9),
                )
            if reason == "stop" and get_value("done_reason") == "length":
                reason = "length"
        logger.info(
            "Tempo total desta chamada: %.1fs (encerrada por: %s)",
            total_duration, reason,
        )
        logger.debug(
            "Início da resposta bruta: %s", response[:300].replace("\n", " ")
        )
        if self.debug:
            self.silver_writer.save_debug_response(image_path, response, debug_id)
        return response, reason

    @staticmethod
    def _validate(response: str) -> ResultadoExtracao:
        """Valida uma resposta completa contra o schema."""
        if not response.strip():
            raise ValueError("O modelo retornou uma resposta vazia.")
        try:
            return ResultadoExtracao.model_validate_json(response)
        except ValidationError as exc:
            raise ValueError(
                f"Resposta do modelo não corresponde ao esquema: {exc}"
            ) from None

    @staticmethod
    def _partial_result(response: str) -> ResultadoExtracao | None:
        """Monta um resultado só com as ofertas completas de uma resposta cortada."""
        data = salvage_offers(response)
        if data is None:
            return None
        try:
            return ResultadoExtracao.model_validate(data)
        except ValidationError:
            logger.debug("Ofertas recuperadas não passaram no schema.", exc_info=True)
            return None

    @staticmethod
    def _offer_signature(offer: dict) -> str:
        """Assinatura usada para deduplicar ofertas entre leituras.

        AJUSTE: normaliza os nomes dos produtos (minúsculas, sem acento) e
        arredonda os preços, para que a mesma oferta lida duas vezes com
        pequenas diferenças ("12.90" vs "12.9", "Arroz 5kg" vs "arroz 5kg")
        seja reconhecida como duplicata.
        """
        variantes = []
        for v in offer.get("variantes") or []:
            nome = normalize_search_text(v.get("produto") or "")
            variantes.append({
                "produto": nome,
                "quantidade": v.get("quantidade"),
                "unidade": v.get("unidade_medida"),
            })
        variantes.sort(key=lambda x: (x["produto"], x["unidade"] or ""))
        return json.dumps({
            "preco": round(float(offer.get("preco_atual") or 0), 2),
            "variantes": variantes,
        }, sort_keys=True, ensure_ascii=False)

    @classmethod
    def _merge(cls, parts: list[ResultadoExtracao | None]) -> ResultadoExtracao | None:
        """Une resultados parciais, removendo ofertas idênticas.

        AJUSTE: a deduplicação usa ``_offer_signature`` (nome normalizado +
        preço arredondado), em vez do JSON exato. Isso evita duplicatas entre
        faixas com sobreposição.
        """
        is_valid = False
        expiry = None
        offers: list[dict] = []
        seen: set[str] = set()
        for part in parts:
            if part is None:
                continue
            data = part.model_dump(mode="json")
            is_valid = is_valid or bool(data.get("encarte_valido"))
            expiry = expiry or data.get("validade")
            for offer in data.get("ofertas") or []:
                signature = cls._offer_signature(offer)
                if signature not in seen:
                    seen.add(signature)
                    offers.append(offer)
        if not offers:
            return None
        try:
            return ResultadoExtracao.model_validate(
                {"encarte_valido": True, "validade": expiry, "ofertas": offers}
            )
        except ValidationError:
            logger.exception("Falha ao unir os resultados parciais.")
            return None

    @staticmethod
    def _needs_reread(
        result: ResultadoExtracao,
        image: Image.Image | None = None,
    ) -> bool:
        """Detecta uma resposta completa, porém provavelmente errada.

        AJUSTE: além dos sinais antigos (variantes demais, produto repetido),
        agora também dispara quando a quantidade de ofertas é muito menor que
        a esperada para o tamanho da imagem (resposta "completa" mas que
        perdeu linhas da grade).
        """
        oversized = any(
            len(offer.variantes) > MAX_SHARED_VARIANTS for offer in result.ofertas
        )
        if oversized or has_repeated_products(result):
            return True
        if image is not None:
            megapixels = (image.width * image.height) / 1_000_000
            esperado = max(1.0, megapixels * MIN_OFFERS_PER_MP)
            if len(result.ofertas) < esperado:
                logger.info(
                    "Cobertura baixa: %d oferta(s) para %.1f MP (esperado ≥%.1f).",
                    len(result.ofertas), megapixels, esperado,
                )
                return True
        return False

    @staticmethod
    def _without_oversized_offers(
        result: ResultadoExtracao,
    ) -> ResultadoExtracao | None:
        """Remove as ofertas com variantes demais, mantendo as demais."""
        data = result.model_dump(mode="json")
        data["ofertas"] = [
            offer for offer in data.get("ofertas") or []
            if len(offer.get("variantes") or []) <= MAX_SHARED_VARIANTS
        ]
        if not data["ofertas"]:
            return None
        try:
            return ResultadoExtracao.model_validate(data)
        except ValidationError:
            return None

    @staticmethod
    def band_boxes(size: tuple[int, int]) -> list[tuple[int, int, int, int]]:
        """Calcula as faixas (com sobreposição) para reler um encarte denso."""
        width, height = size
        vertical = height >= width
        length, thickness = (height, width) if vertical else (width, height)
        band_length = max(int(thickness * BAND_RATIO), MIN_BAND_PX)
        count = min(max(math.ceil(length / band_length), 2), MAX_BANDS)
        step = length / count
        overlap = int(step * BAND_OVERLAP)
        boxes = []
        for index in range(count):
            begin = max(0, int(index * step) - (overlap if index else 0))
            end = min(
                length,
                int((index + 1) * step) + (overlap if index < count - 1 else 0),
            )
            boxes.append(
                (0, begin, width, end) if vertical else (begin, 0, end, height)
            )
        return boxes

    def _split_box(
        self, box: tuple[int, int, int, int]
    ) -> list[tuple[int, int, int, int]]:
        """Divide uma caixa ao meio no lado maior, com sobreposição."""
        x0, y0, x1, y1 = box
        w, h = x1 - x0, y1 - y0
        if h >= w:
            meio = y0 + h // 2
            overlap = int(h * BAND_OVERLAP)
            return [
                (x0, y0, x1, min(y1, meio + overlap)),
                (x0, max(y0, meio - overlap), x1, y1),
            ]
        meio = x0 + w // 2
        overlap = int(w * BAND_OVERLAP)
        return [
            (x0, y0, min(x1, meio + overlap), y1),
            (max(x0, meio - overlap), y0, x1, y1),
        ]

    def _extract_in_bands(
        self,
        image: Image.Image,
        image_path,
        boxes: list[tuple[int, int, int, int]] | None = None,
        depth: int = 0,
    ) -> list[ResultadoExtracao | None]:
        """Relê a imagem em faixas pequenas, com limites curtos de tempo/saída.

        AJUSTE: se uma faixa termina interrompida (loop/timeout/length) e ainda
        há orçamento de profundidade, ela é subdividida em duas e relida. Isso
        garante que nenhuma região fique sem uma segunda chance, o que era a
        principal causa de ofertas perdidas em encartes muito densos.
        """
        if boxes is None:
            boxes = self.band_boxes(image.size)
            logger.warning(
                "Encarte denso: relendo em %d faixas de ~%dx%d.",
                len(boxes), boxes[0][2] - boxes[0][0], boxes[0][3] - boxes[0][1],
            )

        results: list[ResultadoExtracao | None] = []
        for index, box in enumerate(boxes):
            response, reason = self._call_model(
                image.crop(box), image_path, index + 1, len(boxes),
                max_seconds=BAND_MAX_SECONDS,
                num_predict=BAND_NUM_PREDICT,
                debug_number=100 + depth * 100 + index + 1,
            )
            if reason == "stop":
                try:
                    results.append(self._validate(response))
                    continue
                except ValueError as exc:
                    logger.warning("Faixa %d descartada: %s", index + 1, exc)
                    continue

            logger.warning(
                "Faixa %d interrompida (%s); aproveitando o que veio.",
                index + 1, reason,
            )
            partial = self._partial_result(response)
            if partial is not None:
                results.append(partial)

            # AJUSTE: subdivide a faixa que falhou, se ainda houver orçamento.
            x0, y0, x1, y1 = box
            pode_subdividir = (
                depth < BAND_MAX_DEPTH
                and (y1 - y0) > MIN_BAND_PX * 2
                and (x1 - x0) > MIN_BAND_PX * 2
            )
            if pode_subdividir:
                logger.warning(
                    "Subdividindo faixa %d (profundidade %d -> %d).",
                    index + 1, depth, depth + 1,
                )
                results.extend(
                    self._extract_in_bands(
                        image, image_path,
                        boxes=self._split_box(box),
                        depth=depth + 1,
                    )
                )
        return results

    def _reread_suspicious(
        self, result: ResultadoExtracao, image: Image.Image, image_path
    ) -> ResultadoExtracao | None:
        """Refaz a leitura em faixas quando a primeira parece ter agrupado demais."""
        logger.warning(
            "Resultado suspeito em %s; relendo em faixas.", image_path,
        )
        parts = [self._without_oversized_offers(result)]
        parts.extend(self._extract_in_bands(image, image_path))
        return self._merge(parts)

    def extract(
        self,
        image: Image.Image,
        image_path,
        number: int,
        total_count: int,
        attempt: int = 1,
        dense: bool = False,
    ) -> ResultadoExtracao:
        """Consulta o modelo para classificar e extrair ofertas de uma imagem."""
        response, reason = self._call_model(
            image, image_path, number, total_count,
            attempt=attempt, dense=dense,
        )
        if reason == "stop":
            result = self._validate(response)
            # AJUSTE: passa a imagem para a checagem de cobertura.
            if attempt == 1 and not dense and self._needs_reread(result, image):
                reread = self._reread_suspicious(result, image, image_path)
                if reread is not None:
                    return reread
            return result

        logger.warning(
            "Resposta incompleta (%s) em %s; tentando aproveitar.",
            reason, image_path,
        )
        parts = [self._partial_result(response)]
        if attempt == 1 and not dense:
            parts.extend(self._extract_in_bands(image, image_path))
        merged = self._merge(parts)
        if merged is not None:
            logger.warning(
                "Resposta incompleta recuperada: %d oferta(s).",
                len(merged.ofertas),
            )
            return merged
        raise RespostaCortada(
            "Resposta cortada no limite de tokens e nenhuma oferta completa "
            "foi recuperada. Reduza a imagem ou divida o encarte."
        )

    def revisit_products(self, image: Image.Image, image_path) -> list[str]:
        """Confirma por segunda consulta quais produtos de referência aparecem."""
        if not self.segment_products:
            return []
        reference_list = "\n".join(f"- {product}" for product in self.segment_products)
        prompt = PROMPT_PRODUCTS.format(lista=reference_list)
        image_bytes, _ = self.prepare_image(image)
        self.model_used = True
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt, "images": [image_bytes]}],
                format=_RevisitaProdutos.model_json_schema(),
                options=self.build_options(num_predict=768),
                keep_alive=KEEP_ALIVE,
            )
        except Exception:
            logger.exception("Falha na revisita de produtos pelo LLM.")
            return []
        try:
            content = response["message"]["content"]
            data = _RevisitaProdutos.model_validate_json(content)
        except Exception:
            logger.exception("Resposta inválida na revisita de produtos pelo LLM.")
            return []

        items_by_normalized_name = {
            normalize_search_text(product): product
            for product in self.segment_products
        }
        found_products: list[str] = []
        for item in data.confirmados:
            key = normalize_search_text(item.item_da_lista or "")
            if not key:
                continue
            original_name = items_by_normalized_name.get(key)
            if original_name is None:
                continue
            if not item.e_o_proprio_produto or not item.texto_na_imagem.strip():
                logger.info(
                    "Revisita LLM: '%s' recusado (aparece só como sabor/"
                    "ingrediente em '%s').",
                    original_name, item.texto_na_imagem,
                )
                continue
            if original_name not in found_products:
                found_products.append(original_name)
        logger.info(
            "Revisita LLM: %d produto(s) do YAML confirmados na imagem: %s",
            len(found_products), found_products or "(nenhum)",
        )
        return found_products