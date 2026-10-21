"""Integração com o Ollama para classificar e extrair ofertas de imagens.

Este módulo possui o prompt, prepara imagens para inferência, executa as
consultas principal e de revisita e gerencia o processo local do Ollama. O
orquestrador continua responsável pelo fluxo das imagens; ``OllamaService``
fica responsável pela fronteira com o modelo e pelo ciclo de vida do servidor.
"""

from __future__ import annotations

import io
import logging
import os
import subprocess
import time

import ollama
from PIL import Image, ImageDraw
from pydantic import ValidationError

from ..ocr.matching import normalize_search_text
from ..silver_writer import SilverWriter
from .schemas import (
    ProdutosEncontrados,
    RespostaCortada,
    ResultadoExtracao,
    CURRENT_YEAR,
)
from ..settings import (
    MAX_LADO,
    MIN_LADO_ENVIO,
    NUM_BATCH,
    NUM_CTX,
    NUM_PREDICT,
    NUM_THREAD,
    OLLAMA_START_TIMEOUT,
    COMPOSITE_MAX_HEIGHT,
    COMPOSITE_GAP,
)

logger = logging.getLogger(__name__)

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
- produto: nome e variante como escrito.
- marca: identificável ou null.
- quantidade: número sem unidade ("390g"→390, "1,5L"→1.5).
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

VÁRIOS PRODUTOS MESMO PREÇO:
Produtos ligados por "OU", "/", vírgula ou "e" sob um único preço →
uma oferta com várias variantes. Ex.: "IOGURTE NESTLÉ MORANGO OU COCO
170G R$2,49" → 2 variantes (quantidade=170, unidade_medida="g"),
preco_atual=2.49. Vários sabores não nomeados → variante "Sabores".

EXEMPLOS:
1) "ARROZ TIO JOÃO 5KG — R$25,90 / R$22,90 Clube Mais"
   variante: {{produto:"Arroz Tio João", marca:"Tio João",
              quantidade:5, unidade_medida:"kg"}}
   oferta:   {{preco_atual:25.90, preco_condicao:22.90,
              condicao:"Clube Mais"}}

2) "CAFÉ PILÃO 500G — R$18,90"  (só um preço)
   variante: {{produto:"Café Pilão", quantidade:500, unidade_medida:"g"}}
   oferta:   {{preco_atual:18.90, preco_anterior:null,
              preco_condicao:null, condicao:null}}

REGRAS:
1. Não invente dados. Associe cada preço ao produto visualmente correto.
2. Vírgula é decimal: "R$12,99"→12.99; "1,5L"→1.5.
3. Não confunda peso/volume/código/parcelas com preço.
4. "REF:1234" é código, não preço. Ignore "CADA" junto ao preço.
5. Não descarte oferta por campo secundário ilegível.
6. Não duplique ofertas.
7. `condicao` só existe junto de `preco_condicao`. Se um dos dois está
   presente, o outro TAMBÉM deve estar.
"""

PROMPT_PRODUCTS = """
Você recebe uma imagem de encarte de supermercado e uma LISTA DE PRODUTOS
DE REFERÊNCIA. Sua única tarefa é dizer quais itens da lista aparecem na
imagem, mesmo que escritos com variações (marca, tamanho, sabor, plural,
abreviação).

LISTA DE REFERÊNCIA:
{lista}

REGRAS:
1. Devolva APENAS produtos da lista acima, copiados LITERALMENTE como estão
   na lista (não invente, não traduza, não reescreva).
2. Inclua um item se QUALQUER produto visível na imagem for da mesma
   categoria daquele item (ex.: "arroz" na lista casa com "Arroz Tio João
   5kg" na imagem).
3. Se nenhum item da lista aparece, devolva {{"produtos": []}}.
4. Não inclua itens duplicados.
"""


class OllamaService:
    """Executa inferências de visão e gerencia o servidor Ollama.

    Attributes:
        model_name: Nome do modelo selecionado.
        segment_products: Lista de produtos normalizados para a revisita.
        debug: Indica se respostas e imagens devem ser salvas para inspeção.
        model_used: Informa se esta instância já fez uma consulta ao modelo.
    """

    def __init__(
        self,
        model_name: str,
        segment_products: list[str],
        debug: bool,
        silver_writer: SilverWriter,
    ) -> None:
        """Configura o modelo e os recursos usados pelas chamadas.

        Args:
            model_name: Nome de modelo instalado no Ollama.
            segment_products: Produtos usados na chamada de revisita.
            debug: Habilita gravação dos artefatos de cada consulta.
            silver_writer: Componente que persiste os artefatos de debug.
        """
        self.model_name = model_name
        self.segment_products = segment_products
        self.debug = debug
        self.silver_writer = silver_writer
        self.model_used = False
        self._process: subprocess.Popen | None = None

    def start_server_if_needed(self) -> None:
        """Inicia ``ollama serve`` se nenhum servidor responder.

        Raises:
            RuntimeError: Se o executável faltar ou o processo encerrar ao
                iniciar.
            TimeoutError: Se o servidor não ficar disponível no prazo definido.
        """
        try:
            ollama.list()
            logger.info("Ollama já está disponível; usando o servidor existente.")
            return
        except Exception:
            pass

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
        """Encerra somente o processo Ollama iniciado por esta instância.

        Returns:
            ``None``. Falhas de encerramento são registradas no log.
        """
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
        """Encerra processos Ollama preexistentes usando comando do sistema.

        Returns:
            ``None``. Falhas de encerramento são registradas no log.
        """
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
        """Descarrega o modelo ou encerra o processo conforme configuração.

        Args:
            terminate_existing: Se ``True``, encerra servidor preexistente
                quando este serviço usou o modelo.

        Returns:
            ``None``.
        """
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
        """Empilha recortes verticalmente, separados por linha cinza.

        Args:
            crops: Recortes ordenados da imagem original.

        Returns:
            Uma imagem RGB contendo todos os recortes.
        """
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
        """Recorta regiões e divide lotes que excederiam a altura máxima.

        Args:
            image: Imagem original em modo compatível com Pillow.
            regions: Regiões OCR, cada uma contendo uma caixa ``(x0,y0,x1,y1)``.

        Returns:
            Imagens com um ou mais recortes empilhados.
        """
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
        """Redimensiona uma imagem e serializa seus bytes PNG para o modelo.

        Imagens maiores que ``MAX_LADO`` são reduzidas; imagens menores que
        ``MIN_LADO_ENVIO`` são ampliadas. Os metadados incluem dimensões,
        tamanho em KB e estimativa de tokens visuais.

        Args:
            image: Imagem Pillow a enviar ao Ollama.

        Returns:
            Tupla de bytes PNG e metadados da imagem original/enviada.
        """
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

    def extract(
        self,
        image: Image.Image,
        image_path,
        number: int,
        total_count: int,
        attempt: int = 1,
    ) -> ResultadoExtracao:
        """Consulta o modelo para classificar e extrair ofertas de uma imagem.

        Args:
            image: Imagem inteira ou lote de recortes.
            image_path: Caminho original, usado nos logs e artefatos de debug.
            number: Posição da imagem/recorte no lote atual.
            total_count: Quantidade de imagens/recortes no lote atual.
            attempt: Número da tentativa; tentativas extras alteram a
                amostragem para ajudar a escapar de respostas repetitivas.

        Returns:
            Resposta validada pelo modelo ``ResultadoExtracao``.

        Raises:
            RespostaCortada: Se o Ollama interromper a resposta por limite.
            ValueError: Se o conteúdo vier vazio ou não corresponder ao schema.
        """
        image_bytes, info = self.prepare_image(image)
        logger.info(
            "Imagem %d/%d: %dx%d -> %dx%d (%.0f KB, ≈%d tokens de visão)",
            number, total_count, *info["original"], *info["enviada"],
            info["kb"], info["tokens_estimados"],
        )
        if self.debug:
            self.silver_writer.save_debug_image(image_path, image, number)

        start_time = time.monotonic()
        chunks: list[str] = []
        last_chunk = None
        self.model_used = True
        options = {
            "temperature": 0,
            "num_ctx": NUM_CTX,
            "num_predict": NUM_PREDICT,
            "num_thread": NUM_THREAD,
            "num_batch": NUM_BATCH,
        }
        if attempt > 1:
            options.update(
                {"temperature": 0.3, "repeat_penalty": 1.1, "seed": attempt}
            )
        stream = ollama.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": PROMPT, "images": [image_bytes]}],
            format=ResultadoExtracao.model_json_schema(),
            options=options,
            stream=True,
            keep_alive="5m",
        )
        try:
            for chunk in stream:
                if not chunks:
                    logger.info("Primeiro token após %.1fs.", time.monotonic() - start_time)
                chunks.append(chunk["message"]["content"])
                last_chunk = chunk
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
                """Lê um campo do último fragmento do Ollama."""
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
            logger.info("Tempo total desta chamada: %.1fs", total_duration)

        logger.debug(
            "Início da resposta bruta: %s", response[:300].replace("\n", " ")
        )
        if self.debug:
            self.silver_writer.save_debug_response(image_path, response, number)
        if last_chunk is not None:
            reason = (
                last_chunk.get("done_reason") if isinstance(last_chunk, dict)
                else getattr(last_chunk, "done_reason", None)
            )
            if reason == "length":
                raise RespostaCortada(
                    "Resposta cortada no limite de tokens. Reduza a imagem, "
                    "divida o encarte ou aumente NUM_PREDICT/NUM_CTX."
                )
        if not response.strip():
            raise ValueError("O modelo retornou uma resposta vazia.")
        try:
            return ResultadoExtracao.model_validate_json(response)
        except ValidationError as exc:
            raise ValueError(
                f"Resposta do modelo não corresponde ao esquema: {exc}"
            ) from None

    def revisit_products(self, image: Image.Image, image_path) -> list[str]:
        """Confirma por segunda consulta quais produtos de referência aparecem.

        A imagem inteira é enviada para esta etapa. Falhas na requisição ou na
        validação são registradas e retornam uma lista vazia, sem interromper a
        extração principal.

        Args:
            image: Imagem inteira do encarte.
            image_path: Caminho original usado para eventual debug/log.

        Returns:
            Subconjunto, na grafia original, dos produtos da lista de referência.
        """
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
                format=ProdutosEncontrados.model_json_schema(),
                options={
                    "temperature": 0,
                    "num_ctx": NUM_CTX,
                    "num_predict": 512,
                    "num_thread": NUM_THREAD,
                    "num_batch": NUM_BATCH,
                },
                keep_alive="5m",
            )
        except Exception:
            logger.exception("Falha na revisita de produtos pelo LLM.")
            return []
        try:
            content = response["message"]["content"]
            data = ProdutosEncontrados.model_validate_json(content)
        except Exception:
            logger.exception("Resposta inválida na revisita de produtos pelo LLM.")
            return []

        items_by_normalized_name = {
            normalize_search_text(product): product
            for product in self.segment_products
        }
        found_products: list[str] = []
        for item in data.produtos:
            key = normalize_search_text(item or "")
            if not key:
                continue
            original_name = items_by_normalized_name.get(key)
            if original_name and original_name not in found_products:
                found_products.append(original_name)
        logger.info(
            "Revisita LLM: %d produto(s) do YAML confirmados na imagem: %s",
            len(found_products), found_products or "(nenhum)",
        )
        return found_products