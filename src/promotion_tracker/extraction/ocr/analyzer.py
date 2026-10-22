"""Leitura OCR e localização das ofertas em imagens de encartes.

O fluxo desta camada é: inicializar PaddleOCR sob demanda, ler textos e
caixas, localizar produtos da lista de referência e, quando habilitado,
calcular e mesclar regiões que serão enviadas ao modelo. Se a leitura não
for confiável ou não houver produto do segmento, a análise indica ao
orquestrador se deve ignorar a imagem ou enviar o encarte inteiro.
"""

from __future__ import annotations

import logging
import re
import statistics
from dataclasses import dataclass, field
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .matching import normalize_search_text, product_matches
from ..settings import (
    OCR_MIN_PRICES,
    DENSE_OCR_PRICE_THRESHOLD,
    OCR_USE_STRIPS,
    OCR_STRIP_HEIGHT,
    OCR_STRIP_SCALE,
    OCR_STRIP_OVERLAP,
    CROP_PRODUCTS,
    CROP_MAX_HEIGHT_AFTER_PRICE,
    CROP_MAX_PRICE_DISTANCE,
    CROP_BELOW_PRICE_GAP,
    CROP_FIRST_LINE_GAP,
    CROP_NEXT_LINE_GAP,
    CROP_FALLBACK_HEIGHT,
    CROP_FALLBACK_WIDTH,
    CROP_MIN_MERGE_OVERLAP,
    CROP_X_PADDING,
    CROP_Y_PADDING,
    CROP_HEIGHT_RATIO,
    CROP_ALIGNMENT_TOLERANCE,
)

logger = logging.getLogger(__name__)

_RE_PRICE = re.compile(r"(?<!\d)\d{1,4}\s*[.,]\s*\d{2}(?!\d)")
_RE_DIGITS_ONLY = re.compile(r"^\d{1,4}$")
_RE_CENTS = re.compile(r"^[.,]\s*\d{2}$")


def _looks_like_price(text: str) -> bool:
    """Indica se um fragmento do OCR pode fazer parte de um preço.

    Considera preços completos, marcador ``R$``, números isolados e centavos
    isolados, pois o OCR pode separar visualmente cada componente.

    Args:
        text: Conteúdo reconhecido em uma linha.

    Returns:
        ``True`` quando o fragmento parece preço; caso contrário, ``False``.
    """
    text = text.strip()
    if not text:
        return False
    if _RE_PRICE.search(text) or "r$" in text.lower().replace(" ", ""):
        return True
    return bool(_RE_DIGITS_ONLY.match(text) or _RE_CENTS.match(text))


@dataclass
class LinhaOCR:
    """Texto reconhecido pelo OCR e sua caixa delimitadora em pixels."""

    text: str
    normalized_text: str
    x0: float
    y0: float
    x1: float
    y_end: float

    @property
    def width(self) -> float:
        """Largura da caixa delimitadora."""
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        """Altura da caixa delimitadora."""
        return self.y_end - self.y0

    @property
    def center_y(self) -> float:
        """Coordenada vertical do centro da caixa."""
        return (self.y0 + self.y_end) / 2


@dataclass
class RegiaoOferta:
    """Caixa de uma oferta localizada e os produtos associados a ela."""

    box: tuple[int, int, int, int]
    products: list[str]
    method: str
    center: tuple[float, float] | None = None


@dataclass
class AnaliseOCR:
    """Resultado do OCR consumido pelo fluxo de extração.

    Attributes:
        found_products: Produtos do segmento reconhecidos na imagem.
        regions: Regiões de oferta quando a localização foi possível.
        image_size: Dimensões da imagem original como ``(largura, altura)``.
        use_full_image: Indica que não há recortes confiáveis.
        reason: Contexto opcional para logs do orquestrador.
    """

    found_products: list[str]
    regions: list[RegiaoOferta]
    image_size: tuple[int, int]
    use_full_image: bool
    reason: str = ""
    fallback_regions: list[RegiaoOferta] = field(default_factory=list)
    dense: bool = False


class LocalizadorOfertas:
    """Calcula os blocos de oferta com base em texto, preço e posição do OCR.

    Para cada produto, procura um preço próximo e delimita o bloco da oferta
    usando alinhamento de colunas e linhas vizinhas. Se não encontra um preço
    associado, usa uma janela fixa em torno do nome reconhecido.
    """

    def __init__(self, lines: list[LinhaOCR], width: int, height: int) -> None:
        """Prepara estatísticas geométricas para uma imagem.

        Args:
            lines: Linhas de texto reconhecidas com suas posições.
            width: Largura da imagem, em pixels.
            height: Altura da imagem, em pixels.
        """
        self.lines = lines
        self.width = float(width)
        self.height = float(height)
        line_heights = [line.height for line in lines if line.height > 0] or [20.0]
        overall_line_height = statistics.median(line_heights)
        self.precos = [
            index for index, line in enumerate(lines)
            if self._is_price(line, overall_line_height)
        ]
        self._precos_set = set(self.precos)
        text_line_heights = [
            line.height for index, line in enumerate(lines)
            if index not in self._precos_set and line.height > 0
        ]
        self.lh = statistics.median(text_line_heights) if text_line_heights else overall_line_height
        self.tol_x = CROP_ALIGNMENT_TOLERANCE * self.width
        self._pares = self._find_adjacent_pairs()

    @staticmethod
    def _is_price(line: LinhaOCR, overall_line_height: float) -> bool:
        """Classifica uma linha como preço com heurística geométrica.

        Args:
            line: Linha reconhecida pelo OCR.
            overall_line_height: Altura mediana de todas as linhas.

        Returns:
            ``True`` se a linha parece representar um preço.
        """
        text = line.text.strip()
        if _RE_PRICE.search(text):
            return sum(character.isalpha() for character in text) <= 6
        return bool(_RE_DIGITS_ONLY.match(text)) and line.height >= 1.4 * overall_line_height

    def _are_adjacent(self, first_line: LinhaOCR, second_line: LinhaOCR) -> bool:
        """Verifica se duas linhas estão próximas na mesma coluna ou linha.

        Args:
            first_line: Primeira linha candidata.
            second_line: Segunda linha candidata.

        Returns:
            ``True`` quando as caixas são geometricamente adjacentes.
        """
        vertical_gap = second_line.y0 - first_line.y_end
        horizontal_overlap = min(first_line.x1, second_line.x1) - max(
            first_line.x0, second_line.x0
        )
        if (
            second_line.y0 > first_line.y0
            and -0.2 * self.lh <= vertical_gap <= self.lh
            and horizontal_overlap > 0.3 * min(first_line.width, second_line.width)
        ):
            return True
        vertical_overlap = min(first_line.y_end, second_line.y_end) - max(
            first_line.y0, second_line.y0
        )
        horizontal_gap = second_line.x0 - first_line.x1
        return (
            vertical_overlap > 0.5 * min(first_line.height, second_line.height)
            and 0 <= horizontal_gap <= 1.5 * self.lh
        )

    def _find_adjacent_pairs(self) -> list[tuple[int, int]]:
        """Pré-calcula pares de linhas adjacentes para nomes quebrados.

        Returns:
            Índices das linhas vizinhas que podem formar um nome de produto.
        """
        return [
            (i, j)
            for i, first_line in enumerate(self.lines)
            for j, second_line in enumerate(self.lines)
            if i != j and self._are_adjacent(first_line, second_line)
        ]

    def find_occurrences(
        self, products: list[str]
    ) -> list[tuple[str, tuple[int, ...]]]:
        """Encontra as linhas que contêm cada produto de referência.

        Args:
            products: Nomes normalizados dos produtos procurados.

        Returns:
            Pares de produto e índices das linhas correspondentes.
        """
        occurrences: list[tuple[str, tuple[int, ...]]] = []
        for product in products:
            for index, line in enumerate(self.lines):
                if product_matches(product, line.normalized_text):
                    occurrences.append((product, (index,)))
            for first_index, second_index in self._pares:
                first_line = self.lines[first_index]
                second_line = self.lines[second_index]
                combined_text = f"{first_line.normalized_text} {second_line.normalized_text}"
                if (
                    product_matches(product, combined_text)
                    and not product_matches(product, first_line.normalized_text)
                    and not product_matches(product, second_line.normalized_text)
                ):
                    occurrences.append((product, (first_index, second_index)))
        return occurrences

    def _walk_lines(
        self, index: int, direction: int, use_height_limit: bool = False
    ) -> list[int]:
        """Coleta linhas consecutivas alinhadas a uma linha de referência.

        Args:
            index: Índice da linha inicial.
            direction: ``-1`` para cima ou ``1`` para baixo.
            use_height_limit: Interrompe ao encontrar mudança acentuada de altura.

        Returns:
            Índices das linhas pertencentes ao mesmo bloco de texto.
        """
        reference_line = self.lines[index]
        candidates = []
        for current, line in enumerate(self.lines):
            if current == index or current in self._precos_set:
                continue
            if abs(line.x0 - reference_line.x0) > self.tol_x:
                continue
            if direction < 0 and line.y_end <= reference_line.y0 + 0.3 * self.lh:
                candidates.append(current)
            elif direction > 0 and line.y0 >= reference_line.y_end - 0.3 * self.lh:
                candidates.append(current)
        candidates.sort(
            key=lambda current: -self.lines[current].y_end
            if direction < 0 else self.lines[current].y0
        )

        collected_indices: list[int] = []
        edge = reference_line.y0 if direction < 0 else reference_line.y_end
        reference_height = None
        for current in candidates:
            line = self.lines[current]
            gap = edge - line.y_end if direction < 0 else line.y0 - edge
            if not collected_indices:
                deadline = CROP_FIRST_LINE_GAP * self.lh
            elif direction < 0:
                deadline = CROP_NEXT_LINE_GAP * self.lh
            else:
                deadline = CROP_BELOW_PRICE_GAP * self.lh
            if gap > deadline:
                break
            if use_height_limit and reference_height is not None:
                ratio = line.height / reference_height if reference_height else 1.0
                if (
                    ratio > CROP_HEIGHT_RATIO
                    or ratio < 1 / CROP_HEIGHT_RATIO
                ):
                    break
            collected_indices.append(current)
            edge = line.y0 if direction < 0 else line.y_end
            if reference_height is None:
                reference_height = line.height
        return collected_indices

    def _find_associated_price(self, indices: tuple[int, ...]) -> int | None:
        """Localiza o preço mais próximo abaixo das linhas do produto.

        Args:
            indices: Índices das linhas onde o produto foi reconhecido.

        Returns:
            Índice da linha de preço ou ``None`` quando não houver candidato.
        """
        for index in indices:
            if index in self._precos_set:
                return index
        base = [self.lines[index] for index in indices]
        x0 = min(line.x0 for line in base)
        y0 = min(line.y0 for line in base)
        y_end = max(line.y_end for line in base)
        best_index, minimum_distance = None, None
        for price_index in self.precos:
            line = self.lines[price_index]
            if line.y0 < y0 or abs(line.x0 - x0) > 1.5 * self.tol_x:
                continue
            distance = line.y0 - y_end
            if distance > CROP_MAX_PRICE_DISTANCE * self.height:
                continue
            if best_index is None or distance < minimum_distance:
                best_index, minimum_distance = price_index, distance
        return best_index

    def region_for(
        self, product: str, indices: tuple[int, ...]
    ) -> RegiaoOferta:
        """Delimita a área que provavelmente contém uma oferta completa.

        Args:
            product: Produto associado à região.
            indices: Linhas OCR onde o nome foi encontrado.

        Returns:
            Região geométrica da oferta, com janela fixa como fallback.
        """
        width, height = self.width, self.height
        x_padding, y_padding = CROP_X_PADDING * width, CROP_Y_PADDING * height
        base = [self.lines[index] for index in indices]
        price_index = self._find_associated_price(indices)
        if price_index is None:
            return self._fixed_window(product, base)

        price = self.lines[price_index]
        name = [
            self.lines[index]
            for index in self._walk_lines(price_index, -1, use_height_limit=True)
        ]
        block = base + name + [price]
        top = min(line.y0 for line in block) - y_padding
        left = min(line.x0 for line in block) - x_padding
        name_x1 = max(line.x1 for line in block)

        next_items = [
            index for index in self.precos
            if index != price_index
            and self.lines[index].y0 > price.y_end
            and abs(self.lines[index].x0 - price.x0) <= self.tol_x
        ]
        if next_items:
            next_item = min(next_items, key=lambda i: self.lines[i].y0)
            next_name_lines = [
                self.lines[index]
                for index in self._walk_lines(next_item, -1, use_height_limit=True)
            ]
            next_top = min(
                [self.lines[next_item].y0]
                + [line.y0 for line in next_name_lines]
            )
            bottom = next_top - 0.004 * height
        else:
            below_lines = [
                self.lines[index]
                for index in self._walk_lines(price_index, 1)
            ]
            bottom = max([price.y_end] + [line.y_end for line in below_lines]) + y_padding
        bottom = max(bottom, price.y_end + self.lh)
        bottom = min(
            bottom, price.y_end + CROP_MAX_HEIGHT_AFTER_PRICE * height, height
        )

        neighbors = [
            index for index in self.precos
            if index != price_index
            and self.lines[index].x0 - price.x0 > 0.15 * width
            and top <= self.lines[index].center_y <= bottom
        ]
        if neighbors:
            neighbor = min(neighbors, key=lambda i: self.lines[i].x0)
            neighbor_name_lines = [
                self.lines[index].x0
                for index in self._walk_lines(neighbor, -1, use_height_limit=True)
            ]
            limite_direito = min(
                [self.lines[neighbor].x0] + neighbor_name_lines
            ) - 0.03 * width
        else:
            limite_direito = width
        right = max(limite_direito, name_x1 + x_padding, price.x1 + x_padding)

        x0, y0 = max(0.0, left), max(0.0, top)
        x1, y_end = min(width, right), min(height, bottom)
        if (x1 - x0) < 0.10 * width or (y_end - y0) < 2 * self.lh:
            return self._fixed_window(product, base)
        return RegiaoOferta(
            box=(int(x0), int(y0), int(x1), int(y_end)),
            products=[product],
            method="ancora_preco",
            center=self._calculate_center(base),
        )

    @staticmethod
    def _calculate_center(lines: list[LinhaOCR]) -> tuple[float, float]:
        """Calcula o centro médio das caixas de linhas de referência.

        Args:
            lines: Linhas associadas ao produto.

        Returns:
            Coordenadas médias ``(x, y)`` em pixels.
        """
        return (
            sum((line.x0 + line.x1) / 2 for line in lines) / len(lines),
            sum(line.center_y for line in lines) / len(lines),
        )

    def _fixed_window(
        self, product: str, lines: list[LinhaOCR]
    ) -> RegiaoOferta:
        """Cria uma janela ampliada ao redor do nome quando falta preço.

        Args:
            product: Produto reconhecido.
            lines: Linhas que contêm seu nome.

        Returns:
            Região de fallback marcada como ``janela_fixa``.
        """
        x_padding, y_padding = CROP_X_PADDING * self.width, CROP_Y_PADDING * self.height
        x0 = max(0.0, min(line.x0 for line in lines) - x_padding)
        y0 = max(0.0, min(line.y0 for line in lines) - y_padding)
        x1 = min(
            self.width,
            max(line.x1 for line in lines) + x_padding
            + CROP_FALLBACK_WIDTH * self.width,
        )
        y_end = min(
            self.height,
            max(line.y_end for line in lines) + CROP_FALLBACK_HEIGHT * self.height,
        )
        return RegiaoOferta(
            box=(int(x0), int(y0), int(x1), int(y_end)),
            products=[product],
            method="janela_fixa",
            center=self._calculate_center(lines),
        )


def merge_regions(regions: list[RegiaoOferta]) -> list[RegiaoOferta]:
    """Mescla recortes sobrepostos e descarta janelas internas a uma oferta.

    Janelas fixas contidas dentro de um bloco ancorado em preço são removidas
    para evitar tratar texto impresso em uma embalagem como nova oferta.

    Args:
        regions: Regiões independentes detectadas no encarte.

    Returns:
        Regiões mescladas e ordenadas de cima para baixo.
    """
    anchors = [region.box for region in regions if region.method == "ancora_preco"]

    def is_inside_block(region: RegiaoOferta) -> bool:
        """Verifica se o centro da região está dentro de um bloco ancorado."""
        if region.center is None:
            return False
        center_x, center_y = region.center
        return any(
            x0 <= center_x <= x1 and y0 <= center_y <= y_end
            for x0, y0, x1, y_end in anchors
        )

    pending_items = [
        region for region in regions
        if not (
            region.method == "janela_fixa" and is_inside_block(region)
        )
    ]
    merged_regions: list[RegiaoOferta] = []
    while pending_items:
        current = pending_items.pop(0)
        changed = True
        while changed:
            changed = False
            for other_item in list(pending_items):
                ax0, ay0, ax1, ay1 = current.box
                bx0, by0, bx1, by1 = other_item.box
                intersection_width = min(ax1, bx1) - max(ax0, bx0)
                intersection_height = min(ay1, by1) - max(ay0, by0)
                menor_area = min(
                    (ax1 - ax0) * (ay1 - ay0),
                    (bx1 - bx0) * (by1 - by0),
                )
                if (
                    intersection_width > 0
                    and intersection_height > 0
                    and intersection_width * intersection_height
                    >= CROP_MIN_MERGE_OVERLAP * menor_area
                ):
                    current = RegiaoOferta(
                        box=(
                            min(ax0, bx0), min(ay0, by0),
                            max(ax1, bx1), max(ay1, by1),
                        ),
                        products=sorted(set(current.products) | set(other_item.products)),
                        method=(
                            "janela_fixa"
                            if "janela_fixa" in (current.method, other_item.method)
                            else "ancora_preco"
                        ),
                    )
                    pending_items.remove(other_item)
                    changed = True
        merged_regions.append(current)
    merged_regions.sort(key=lambda region: (region.box[1], region.box[0]))
    return merged_regions


class OCRAnalyzer:
    """Analisa uma imagem e retorna produtos e recortes confiáveis.

    PaddleOCR é importado e inicializado apenas no primeiro uso. Se não houver
    correspondências do segmento, ``analisar`` retorna ``None`` e registra
    ``motivo_ignorada`` para o extrator persistir a decisão.
    """

    def __init__(
        self,
        segment_products: list[str],
        debug: bool = False,
        product_aliases: dict[str, list[str]] | None = None,
    ) -> None:
        """Configura o analisador sem carregar o motor OCR antecipadamente.

        Args:
            segment_products: Produtos normalizados que devem ser procurados.
            debug: Habilita amostras detalhadas de texto nos logs.
            product_aliases: Termos alternativos mapeados para cada produto.
        """
        self.segment_products = segment_products
        self.product_aliases = product_aliases or {
            product: [product] for product in segment_products
        }
        self._alias_to_product = {
            normalize_search_text(alias): product
            for product, aliases in self.product_aliases.items()
            for alias in aliases
        }
        self.debug = debug
        self._ocr_reader = None
        self.ignored_reason = ""

    def _load_ocr(self) -> None:
        """Inicializa PaddleOCR uma única vez para esta instância."""
        if self._ocr_reader is not None:
            return
        from paddleocr import PaddleOCR

        self._ocr_reader = PaddleOCR(
            lang="pt",
            device="cpu",
            engine="onnxruntime",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )

    @staticmethod
    def _get_result_data(result: object) -> dict:
        """Normaliza os formatos de resultado suportados por PaddleOCR.

        Args:
            result: Objeto retornado por ``PaddleOCR.predict``.

        Returns:
            Conteúdo interno com textos e posições ou dicionário vazio.
        """
        data = None
        try:
            data = result.json
            if callable(data):
                data = data()
        except Exception:
            data = None
        if data is None and isinstance(result, dict):
            data = result
        if not isinstance(data, dict):
            return {}
        interno = data.get("res")
        return interno if isinstance(interno, dict) else data

    @staticmethod
    def _extract_boxes(result_data: dict, count: int) -> list[tuple] | None:
        """Extrai caixas de ``rec_boxes`` ou converte polígonos em retângulos.

        Args:
            result_data: Dados normalizados de um resultado PaddleOCR.
            count: Número de textos reconhecidos que precisam de caixa.

        Returns:
            Caixas ``(x0, y0, x1, y1)`` ou ``None`` se as posições faltarem.
        """
        boxes = result_data.get("rec_boxes")
        if boxes is not None and len(boxes) == count:
            return [tuple(float(value) for value in box[:4]) for box in boxes]
        for key in ("rec_polys", "dt_polys"):
            polygons = result_data.get(key)
            if polygons is not None and len(polygons) == count:
                converted_boxes = []
                for polygon in polygons:
                    xs = [float(point[0]) for point in polygon]
                    ys = [float(point[1]) for point in polygon]
                    converted_boxes.append((min(xs), min(ys), max(xs), max(ys)))
                return converted_boxes
        return None

    def _read_ocr(
        self, rgb_image: np.ndarray
    ) -> tuple[list[str], list[LinhaOCR], bool]:
        """Executa o OCR e coleta textos e posições para uma imagem RGB.

        Args:
            rgb_image: Imagem em array NumPy, com canais RGB.

        Returns:
            Tupla com textos, linhas posicionadas e indicador de posições
            disponíveis para todos os resultados.
        """
        texts: list[str] = []
        lines: list[LinhaOCR] = []
        positions_available = True
        for result in self._ocr_reader.predict(rgb_image):
            result_data = self._get_result_data(result)
            if not result_data:
                logger.warning(
                    "Resultado do OCR em formato inesperado (tipo %s).",
                    type(result).__name__,
                )
                continue
            recognized_texts = [str(text) for text in list(result_data.get("rec_texts", []))]
            boxes = self._extract_boxes(result_data, len(recognized_texts))
            texts.extend(recognized_texts)
            if boxes is None:
                positions_available = False
                continue
            for text, (x0, y0, x1, y_end) in zip(recognized_texts, boxes):
                lines.append(LinhaOCR(
                    text=text,
                    normalized_text=normalize_search_text(text),
                    x0=x0, y0=y0, x1=x1, y_end=y_end,
                ))
        return texts, lines, positions_available

    def _read_ocr_strips(self, rgb_image: np.ndarray) -> list[str]:
        """Relê a imagem em faixas horizontais ampliadas e sobrepostas.

        Args:
            rgb_image: Imagem original em array NumPy RGB.

        Returns:
            Textos identificados nas faixas.
        """
        height = rgb_image.shape[0]
        step = max(1, int(OCR_STRIP_HEIGHT * (1 - OCR_STRIP_OVERLAP)))
        texts: list[str] = []
        y_position = 0
        while True:
            y_end = min(height, y_position + OCR_STRIP_HEIGHT)
            strip = rgb_image[y_position:y_end]
            if OCR_STRIP_SCALE != 1.0:
                image = Image.fromarray(strip)
                image = image.resize(
                    (
                        round(image.width * OCR_STRIP_SCALE),
                        round(image.height * OCR_STRIP_SCALE),
                    ),
                    Image.Resampling.LANCZOS,
                )
                strip = np.array(image)
            texts.extend(self._read_ocr(strip)[0])
            if y_end >= height:
                break
            y_position += step
        return texts

    def _find_products(self, texts: list[str]) -> list[str]:
        """Compara o texto OCR com a lista normalizada do segmento.

        Args:
            texts: Fragmentos de texto reconhecidos.

        Returns:
            Produtos de referência encontrados, sem alterar sua grafia.
        """
        ocr_text = normalize_search_text(" ".join(texts))
        return [
            product for product in self.segment_products
            if any(
                product_matches(
                    normalize_search_text(alias),
                    ocr_text,
                )
                for alias in self.product_aliases.get(product, [product])
            )
        ]

    def analyze(self, image_path: Path) -> AnaliseOCR | None:
        """Lê uma imagem, decide se deve seguir ao modelo e calcula regiões.

        Args:
            image_path: Caminho da imagem original.

        Returns:
            Análise OCR quando há produto de referência; ``None`` quando a
            imagem deve ser ignorada antes de chamar o modelo.
        """
        self._load_ocr()
        with Image.open(image_path) as image:
            rgb_image = np.array(image.convert("RGB"))
        height, width = rgb_image.shape[:2]
        self.ignored_reason = "sem_produto_do_segmento"
        texts, lines, positions_available = self._read_ocr(rgb_image)
        logger.info("OCR leu %d linhas de texto.", len(texts))
        if self.debug:
            logger.debug("Amostra do OCR: %s", texts[:20])

        found_products = self._find_products(texts)
        used_strips = False
        if not found_products and OCR_USE_STRIPS:
            logger.info(
                "Nenhum produto na leitura inteira; relendo em faixas ampliadas..."
            )
            strip_texts = self._read_ocr_strips(rgb_image)
            logger.info("OCR em faixas leu %d linhas de texto.", len(strip_texts))
            texts += strip_texts
            found_products = self._find_products(texts)
            used_strips = bool(found_products)
        if not found_products:
            logger.info(
                "Nenhum produto do segmento encontrado pelo OCR. Amostra: %s",
                normalize_search_text(" ".join(texts))[:200] or "(vazio)",
            )
            return None

        logger.info(
            "Produtos do segmento encontrados pelo OCR%s: %s",
            " (leitura em faixas)" if used_strips else "",
            ", ".join(found_products),
        )
        price_count = sum(1 for text in texts if _looks_like_price(text))
        logger.info("Preços detectados pelo OCR: %d", price_count)
        if OCR_MIN_PRICES and price_count < OCR_MIN_PRICES:
            logger.info(
                "Poucos preços no OCR (%d < %d): não parece encarte.",
                price_count, OCR_MIN_PRICES,
            )
            self.ignored_reason = "poucos_precos_no_ocr"
            return None

        image_size = (width, height)
        if used_strips:
            return AnaliseOCR(found_products, [], image_size, True,
                              "produto achado na leitura em faixas")
        if not positions_available or not lines:
            logger.warning(
                "OCR sem posições confiáveis. Enviando o encarte inteiro."
            )
            return AnaliseOCR(found_products, [], image_size, True, "sem posições do OCR")

        localizador = LocalizadorOfertas(lines, width, height)
        logger.debug(
            "Geometria: altura típica de linha=%.0fpx | preços detectados=%d",
            localizador.lh, len(localizador.precos),
        )
        search_terms = [
            normalize_search_text(alias)
            for product in found_products
            for alias in self.product_aliases.get(product, [product])
        ]
        occurrences = localizador.find_occurrences(search_terms)
        regions = []
        for term, indices in occurrences:
            product = self._alias_to_product.get(term, term)
            regions.append(localizador.region_for(product, indices))
        located_products = {product for region in regions for product in region.products}
        products_without_region = [
            product for product in found_products if product not in located_products
        ]
        if products_without_region:
            logger.warning(
                "Não foi possível localizar na imagem: %s. Enviando encarte inteiro.",
                ", ".join(products_without_region),
            )
            return AnaliseOCR(found_products, [], image_size, True, "produto não localizado")

        regions = merge_regions(regions)
        for number, region in enumerate(regions, start=1):
            logger.info(
                "Região %d: %s | caixa=%s | método=%s",
                number, ", ".join(region.products), region.box, region.method,
            )
        if not CROP_PRODUCTS:
            return AnaliseOCR(
                found_products,
                [],
                image_size,
                True,
                "recorte desligado",
                fallback_regions=regions,
                dense=price_count >= DENSE_OCR_PRICE_THRESHOLD,
            )
        return AnaliseOCR(found_products, regions, image_size, False)