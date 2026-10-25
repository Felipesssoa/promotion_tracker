"""Prompt, schema e parsing da leitura por recorte.

O prompt e o schema são os do extrator original (classificação rigorosa por
preço, variantes, preço de condição, validade em ISO), com acréscimos para
leitura de recortes e para marcas. A resposta é validada pelos mesmos modelos
Pydantic do projeto (``ResultadoExtracao``), então o resultado cai no
``SilverWriter`` sem conversões.
"""
from __future__ import annotations

import json
import re
from collections import Counter

from promotion_tracker.extraction.llm.schemas import CURRENT_YEAR, Oferta, ResultadoExtracao

from .brands import BrandCatalog

TILE_SCHEMA = ResultadoExtracao.model_json_schema()

_PROMPT_TEMPLATE = """
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
- A imagem é um RECORTE do encarte: pode mostrar só uma parte dele, e itens
  podem estar cortados nas bordas. Extraia só ofertas COMPLETAS (nome e preço
  visíveis); ignore itens cortados na borda (aparecem inteiros em outro recorte).
- Cabeçalho, rodapé, logo ou área sem produto e preço: encarte_valido=false.

SE ENCARTE, extraia TODAS as ofertas do recorte, sem pular nenhuma, mesmo com
texto pequeno ou layout denso. Cada oferta precisa de um preço visível.

VALIDADE (campo "validade"):
- Procure na imagem por textos como "Válido até", "Validade:",
  "Ofertas válidas de X a Y", "Promoção válida até".
- Devolva SEMPRE a data FINAL no formato ISO "YYYY-MM-DD" (ex.: "2026-10-25").
- Se a imagem trouxer só dia/mês ("até 25/10") use o ano corrente
  ({ANO}); se essa data já passou há mais de 30 dias, use {ANO_SEGUINTE}.
- Se a imagem trouxer mês por extenso ("25 de outubro de 2026"), converta
  para ISO.
- Se a imagem trouxer APENAS um dia da semana ("terça-feira") sem data,
  devolva null (não é data absoluta).
- Se não houver validade na imagem, devolva null.

VARIANTE (por produto):
- produto: nome e variante como escrito, mantendo abreviações (REQ, QJ, UHT).
- marca: SEMPRE que a marca puder ser identificada, preencha. Procure em três
  lugares: (1) no próprio nome ("Leite Betânia" -> Betânia), (2) no logotipo ou
  na embalagem mostrada junto ao produto, (3) em texto de marca perto do item.
  Quando for uma das marcas conhecidas abaixo, escreva exatamente como na lista.
  Só use null se realmente não houver nenhuma pista de marca.
  Marcas conhecidas: {MARCAS}
- quantidade: número sem unidade ("390g"->390, "1,5L"->1.5).
- unidade_medida: "g"|"kg"|"mg"|"ml"|"l"|"un"|"outro"|null.
- Item vendido a peso (frutas, carnes, frios): quantidade=1, unidade_medida="kg",
  preço por kg.

OFERTA (por preço):
- preco_atual: preço principal (normal) em reais, decimal. Sempre presente.
- preco_anterior: preço anterior explícito ("de R$ X por R$ Y" -> X) ou null.
- preco_condicao: SEGUNDO preço da oferta, quando aparece com uma
  CONDIÇÃO (Clube, App, cartão, "leve 3 pague 2", "só para membros",
  "clientes cadastrados", atacado a partir de N unidades).
  É o preço menor, impresso logo abaixo/ao lado do principal, com uma
  etiqueta colorida ou em negrito da condição. Ex. na tela:
      "R$ 25,90"
      "R$ 22,90 no Clube Mais"
  -> preco_atual=25.90, preco_condicao=22.90, condicao="Clube Mais".
  Se houver SÓ um preço, preco_condicao=null e condicao=null.
- condicao: texto EXATO da etiqueta da condição, como está escrito
  ("Clube Mais", "App", "Cartão Fidelidade", "leve 3 pague 2"). SEMPRE
  preencha se houver preco_condicao. SEMPRE deixe null se não houver
  segundo preço com condição.
- Muitos encartes escrevem o preço com o número grande e os centavos pequenos,
  no alto, ao lado (5 grande e 99 pequeno = 5.99). Junte as duas partes.
- Item embalado: ignore o preço por kg ou por litro em letra miúda (preço de
  comparação).

VÁRIOS PRODUTOS MESMO PREÇO:
Produtos ligados por "OU", "/", vírgula ou "e" sob um único preço ->
uma oferta com várias variantes. Ex.: "IOGURTE NESTLÉ MORANGO OU COCO
170G R$2,49" -> 2 variantes (quantidade=170, unidade_medida="g"),
preco_atual=2.49. Vários sabores não nomeados -> variante "Sabores".

EXEMPLOS:
1) "ARROZ TIO JOÃO 5KG — R$25,90 / R$22,90 Clube Mais"
   variante: {produto:"Arroz Tio João", marca:"Tio João",
              quantidade:5, unidade_medida:"kg"}
   oferta:   {preco_atual:25.90, preco_condicao:22.90,
              condicao:"Clube Mais"}

2) "CAFÉ PILÃO 500G — R$18,90"  (só um preço)
   variante: {produto:"Café Pilão", marca:"Pilão",
              quantidade:500, unidade_medida:"g"}
   oferta:   {preco_atual:18.90, preco_anterior:null,
              preco_condicao:null, condicao:null}

REGRAS:
1. Não invente dados. Associe cada preço ao produto visualmente correto
   (dentro do mesmo bloco; não troque preços de itens vizinhos).
2. Vírgula é decimal: "R$12,99"->12.99; "1,5L"->1.5.
3. Não confunda peso/volume/código/parcelas com preço.
4. "REF:1234" é código, não preço. Ignore "CADA" junto ao preço.
5. Não descarte oferta por campo secundário ilegível (marca, quantidade).
6. Não duplique ofertas.
7. `condicao` só existe junto de `preco_condicao`. Se um dos dois está
   presente, o outro TAMBÉM deve estar.
"""


def build_prompt(brands: BrandCatalog | None = None) -> str:
    brand_list = brands.prompt_list() if brands and brands.official else "(nenhuma lista)"
    return (
        _PROMPT_TEMPLATE
        .replace("{ANO_SEGUINTE}", str(CURRENT_YEAR + 1))
        .replace("{ANO}", str(CURRENT_YEAR))
        .replace("{MARCAS}", brand_list)
    )


PROMPT = build_prompt()


# --------------------------------------------------------------------------- parsing
def _salvage(text: str) -> ResultadoExtracao:
    """Recupera as ofertas completas de um JSON truncado (resposta cortada)."""
    anchor = text.find('"ofertas"')
    start = text.find("[", anchor if anchor != -1 else 0)
    offers: list[Oferta] = []
    if start != -1:
        decoder, index = json.JSONDecoder(), start + 1
        while index < len(text):
            while index < len(text) and text[index] in " \n\r\t,":
                index += 1
            if index >= len(text) or text[index] != "{":
                break
            try:
                obj, index = decoder.raw_decode(text, index)
                offers.append(Oferta.model_validate(obj))
            except (json.JSONDecodeError, ValueError):
                break
    match = re.search(r'"validade"\s*:\s*"([^"]*)"', text)
    return ResultadoExtracao(
        encarte_valido=bool(offers),
        validade=match.group(1) if match else None,
        ofertas=offers,
    )


def parse_tile_response(text: str) -> ResultadoExtracao:
    """Valida a resposta do modelo. Levanta ValueError se vier vazia."""
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    if not cleaned:
        raise ValueError("resposta vazia")

    try:
        return ResultadoExtracao.model_validate_json(cleaned)
    except ValueError:
        pass

    first, last = cleaned.find("{"), cleaned.rfind("}")
    if first != -1 and last > first:
        try:
            return ResultadoExtracao.model_validate(json.loads(cleaned[first : last + 1]))
        except ValueError:
            pass
    return _salvage(cleaned)


def fill_missing_brands(result: ResultadoExtracao, brands: BrandCatalog) -> int:
    """Preenche `marca` ausente quando o nome do produto cita uma marca conhecida."""
    filled = 0
    for offer in result.ofertas:
        for variant in offer.variantes:
            if not variant.marca:
                found = brands.find_in_text(variant.produto)
                if found:
                    variant.marca = found
                    filled += 1
    return filled


def most_common(values: list[str | None]) -> str | None:
    counted = Counter(v for v in values if v)
    return counted.most_common(1)[0][0] if counted else None