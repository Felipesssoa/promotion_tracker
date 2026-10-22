"""Orquestra a extração de ofertas de imagens da camada Bronze.

Fluxo principal:
    1. ``LLMExtractor`` carrega configuração, produtos de referência e Ollama.
    2. ``OCRAnalyzer`` lê cada imagem e localiza produtos e regiões candidatas.
    3. O Ollama classifica a imagem e extrai ofertas validadas por Pydantic.
    4. Uma segunda consulta confirma produtos de referência na imagem inteira.
    5. O extrator expande as variantes e grava o resultado na camada Silver.

As responsabilidades de configuração, correspondência de produtos, OCR e
contratos de dados ficam em módulos próprios. Este módulo mantém a API pública
``LLMExtractor`` e coordena o processamento em lote, chamadas ao modelo,
persistência e encerramento dos recursos.
"""

import atexit
import logging
import os
import signal
import time
from pathlib import Path

from PIL import Image

from ..common.logging_config import configure_logging
from ..common.paths import PROJECT_PATHS
from ..common.products import ProductCatalog
from .llm.schemas import RespostaCortada, ResultadoExtracao
from .llm.service import OllamaService
from .ocr.analyzer import AnaliseOCR, OCRAnalyzer, RegiaoOferta
from .ocr.matching import FUZZY_AVAILABLE, normalize_search_text
from .settings import (
    CROP_PRODUCTS,
    DENSE_NUM_CTX,
    DENSE_NUM_PREDICT,
    FUZZY_THRESHOLD,
    MANAGE_OLLAMA_SERVER,
    MATCH_FUZZY,
    MATCH_JOINED_TEXT,
    MAX_LADO,
    NUM_BATCH,
    NUM_CTX,
    NUM_PREDICT,
    NUM_THREAD,
    TERMINATE_EXISTING_SERVER,
)
from .silver_writer import SilverWriter

# ----------------------------------------------------------------------
# Variáveis de ambiente recomendadas (definidas antes do ollama importar)
# ----------------------------------------------------------------------
# os.environ.setdefault("OLLAMA_FLASH_ATTENTION", "1")
# os.environ.setdefault("OLLAMA_KV_CACHE_TYPE", "q8_0")


logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Validade: parsing e normalização
# ----------------------------------------------------------------------
def _configure_logging(debug: bool) -> None:
    """
    Configura o logging da aplicação

    Args:
        debug: Se True, eleva o nível para DEBUG

    Returns:
        None
    """
    configure_logging(debug=debug)


# ----------------------------------------------------------------------
# Extrator
# ----------------------------------------------------------------------

class LLMExtractor:
    """
    Processa imagens da camada Bronze com Qwen2.5-VL via Ollama

    O OCR acha os produtos do segmento e o encarte é recortado só nas
    regiões desses produtos. O modelo classifica e extrai na mesma resposta,
    validada com Pydantic. Depois, uma segunda chamada ao LLM revisita a
    imagem inteira para casar os produtos do product.yaml com o que aparece
    nela e preencher o campo `produtos_encontrados`

    Encerra o servidor Ollama ao final (fim do lote, Ctrl+C, SIGTERM/SIGHUP,
    fechamento do terminal no Windows, atexit)

    Ordem de execução das funções:
        __init__() -> process_bronze_folder() -> _process_bronze_folder() ->
        para cada imagem: _find_segment_products() -> _extract_data() ->
        _query_model() -> _revisit_segment_products() -> _save_json()
    """

    IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}

    def __init__(
        self,
        model_name: str = "qwen2.5vl:latest",
        use_ocr_images: bool = False,
        skip_existing: bool = True,
        debug: bool = False,
        max_images: int | None = None,
    ) -> None:
        """
        Inicializa o extrator e prepara as pastas e o servidor Ollama

        Args:
            model_name: Nome do modelo no Ollama
            use_ocr_images: Se True, processa imagens terminadas em "_ocr"
            skip_existing: Se True, ignora imagens já processadas (JSON existe)
            debug: Se True, salva a resposta bruta e a imagem enviada
            max_images: Limite opcional de imagens por execução
        """
        _configure_logging(debug)

        self.base_dir = PROJECT_PATHS.root
        self.product_catalog = ProductCatalog()
        self.bronze_dir = PROJECT_PATHS.bronze_dir
        self.silver_dir = PROJECT_PATHS.silver_dir
        self.segment_products = self.product_catalog.products
        self._ocr_analyzer = OCRAnalyzer(
            self.segment_products,
            debug=debug,
            product_aliases=self.product_catalog.search_terms_by_product,
        )

        self.model_name = model_name
        self._silver_writer = SilverWriter(
            self.bronze_dir,
            self.silver_dir,
            self.model_name,
            product_catalog=self.product_catalog,
        )
        self._ollama_service = OllamaService(
            self.model_name, self.segment_products, debug, self._silver_writer
        )
        self.use_ocr_images = use_ocr_images
        self.skip_existing = skip_existing
        self.debug = debug
        self.max_images = max_images

        self._stats = self._new_stats()
        self._fechando = False
        self._fechado = False
        self._console_ctrl_handler = None
        self._motivo_ignorada = ""

        if not self.bronze_dir.exists():
            raise FileNotFoundError(
                f"Pasta Bronze não encontrada: {self.bronze_dir}"
            )
        self.silver_dir.mkdir(parents=True, exist_ok=True)

        self._register_exit_handlers()

        if MANAGE_OLLAMA_SERVER:
            self._start_server_if_needed()

        logger.info("Modelo: %s | debug: %s", self.model_name, self.debug)
        logger.info(
            "Extração: NUM_CTX=%d | NUM_PREDICT=%d | MAX_LADO=%d px",
            NUM_CTX, NUM_PREDICT, MAX_LADO,
        )
        logger.info(
            "CPU: NUM_THREAD=%d (físicos) | NUM_BATCH=%d",
            NUM_THREAD, NUM_BATCH,
        )
        logger.info(
            "Recorte por produto: %s",
            "ligado" if CROP_PRODUCTS else "desligado (encarte inteiro)",
        )
        logger.info(
            "Casamento: colado=%s | fuzzy=%s (limiar %d)",
            MATCH_JOINED_TEXT,
            "ligado" if (MATCH_FUZZY and FUZZY_AVAILABLE)
            else "desligado (instale rapidfuzz)" if MATCH_FUZZY
            else "desligado",
            FUZZY_THRESHOLD,
        )
        logger.info(
            "Produtos do segmento (%d): %s",
            len(self.segment_products), self.segment_products,
        )
        logger.info("Bronze: %s", self.bronze_dir)
        logger.info("Silver: %s", self.silver_dir)

    @staticmethod
    def _new_stats() -> dict:
        """
        Cria um dicionário de estatísticas zerado

        Returns:
            dict: Estatísticas de ofertas, produtos, compartilhadas e avisos
        """
        return {"ofertas": 0, "produtos": 0, "compartilhadas": 0, "avisos": 0}

    # ------------------------------------------------------------------
    # Servidor Ollama
    # ------------------------------------------------------------------

    def _start_server_if_needed(self) -> None:
        """
        Inicia o "ollama serve" caso ainda não esteja disponível

        Raises:
            RuntimeError: Se o executável não for encontrado ou o processo
                encerrar durante a inicialização
            TimeoutError: Se o servidor não ficar disponível no prazo
        """
        self._ollama_service.start_server_if_needed()

    def _stop_owned_server(self) -> None:
        """
        Encerra o servidor Ollama iniciado por esta instância

        Returns:
            None
        """
        self._ollama_service.stop_owned_server()

    def _stop_existing_server(self) -> None:
        """
        Encerra processos Ollama já em execução no sistema

        Returns:
            None
        """
        self._ollama_service.stop_existing_servers()

    # ------------------------------------------------------------------
    # Handlers de saída
    # ------------------------------------------------------------------

    def _register_exit_handlers(self) -> None:
        """
        Registra handlers para encerrar o servidor em qualquer saída

        Returns:
            None
        """
        for sig_name in ("SIGINT", "SIGTERM", "SIGHUP", "SIGBREAK"):
            sig = getattr(signal, sig_name, None)
            if sig is None:
                continue
            try:
                signal.signal(sig, self._handle_signal)
            except (ValueError, OSError):
                pass
        try:
            atexit.register(self.close)
        except Exception:
            logger.debug("Falha ao registrar atexit.", exc_info=True)
        if os.name == "nt":
            self._register_windows_console_close()

    def _register_windows_console_close(self) -> None:
        """
        Registra o handler de fechamento do console no Windows

        Returns:
            None
        """
        try:
            import ctypes
            from ctypes import wintypes
            kernel32 = ctypes.windll.kernel32
            HandlerRoutine = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.DWORD)
            extractor_ref = self

            @HandlerRoutine
            def handler(event_type):
                """Libera os recursos quando o console é fechado."""
                if event_type in (2, 5, 6):
                    try:
                        extractor_ref.close()
                    except Exception:
                        pass
                    return True
                return False

            kernel32.SetConsoleCtrlHandler(handler, True)
            self._console_ctrl_handler = handler
        except Exception:
            logger.debug(
                "Falha ao registrar handler de console.", exc_info=True
            )

    def _handle_signal(self, signum, frame) -> None:
        """
        Trata sinais de encerramento (Ctrl+C, SIGTERM, etc.)

        Args:
            signum: Número do sinal recebido
            frame: Frame atual (não utilizado)

        Returns:
            None
        """
        if self._fechando or self._fechado:
            raise KeyboardInterrupt
        name = getattr(signal.Signals(signum), "name", str(signum))
        logger.warning("Sinal %s recebido. Liberando recursos...", name)
        try:
            self.close()
        except Exception:
            logger.exception("Erro ao liberar recursos no handler de sinal.")
        if signum == getattr(signal, "SIGINT", None):
            raise KeyboardInterrupt
        raise SystemExit(128 + signum)

    # ------------------------------------------------------------------
    # Descoberta
    # ------------------------------------------------------------------

    def _json_path(self, image_path: Path) -> Path:
        """
        Calcula o caminho do JSON de saída correspondente a uma imagem

        Args:
            image_path: Caminho da imagem na Bronze

        Returns:
            Path: Caminho do JSON na Silver
        """
        return self._silver_writer.json_path(image_path)

    def _is_candidate(self, path: Path) -> bool:
        """
        Verifica se um arquivo é uma imagem a ser processada

        Args:
            path: Caminho do arquivo

        Returns:
            bool: True se for candidato à extração
        """
        if not path.is_file() or path.suffix.lower() not in self.IMAGE_EXTENSIONS:
            return False
        if path.stem.lower().endswith(".tmp"):
            return False
        is_ocr_image = path.stem.lower().endswith("_ocr")
        return is_ocr_image == self.use_ocr_images

    @staticmethod
    def _normalize_search_text(text: str) -> str:
        """
        Wrapper estático para normalizar texto de busca

        Args:
            text: Texto original

        Returns:
            str: Texto normalizado
        """
        return normalize_search_text(text)

    # ------------------------------------------------------------------
    # OCR
    # ------------------------------------------------------------------

    def _find_segment_products(self, image_path: Path) -> AnaliseOCR | None:
        """Delegates OCR and region detection to the dedicated analyzer.

        Args:
            image_path: Caminho da imagem original.

        Returns:
            Análise para imagens com produtos do segmento, ou ``None`` para
            imagens descartadas antes da chamada ao modelo.
        """
        analysis = self._ocr_analyzer.analyze(image_path)
        self._motivo_ignorada = self._ocr_analyzer.ignored_reason
        return analysis

    # ------------------------------------------------------------------
    # Lote
    # ------------------------------------------------------------------

    def process_bronze_folder(self) -> None:
        """
        Processa todas as imagens da Bronze e encerra o servidor ao final

        Returns:
            None
        """
        try:
            self._process_bronze_folder()
        finally:
            self.close()

    def _process_bronze_folder(self) -> None:
        """
        Percorre as imagens da Bronze e processa cada uma

        Returns:
            None
        """
        batch_start_time = time.monotonic()
        self._stats = self._new_stats()

        images = sorted(
            path for path in self.bronze_dir.rglob("*")
            if self._is_candidate(path)
        )

        if not images:
            item_type = (
                "terminadas em '_ocr'"
                if self.use_ocr_images else "originais"
            )
            logger.warning(
                "Nenhuma imagem %s em %s", item_type, self.bronze_dir
            )
            return

        logger.info("Imagens encontradas na Bronze: %d", len(images))

        if self.skip_existing:
            pending_items = [
                p for p in images if not self._json_path(p).exists()
            ]
            logger.info(
                "Já processadas: %d | Pendentes: %d",
                len(images) - len(pending_items), len(pending_items),
            )
            images = pending_items

        if self.max_images is not None:
            images = images[: self.max_images]
            logger.info(
                "Limitando a %d imagem(ns) (max_images).", len(images)
            )

        if not images:
            logger.info("Nada novo para processar.")
            return

        successes = skipped_by_ocr = failures = 0
        durations: list[float] = []

        for index, image_path in enumerate(images, start=1):
            profile_name = image_path.parent.name
            start_time = time.monotonic()

            logger.info("=" * 60)
            logger.info(
                "[%d/%d] %s | perfil: %s",
                index, len(images), image_path.name, profile_name,
            )

            try:
                analysis = self._find_segment_products(image_path)
                if analysis is None:
                    logger.info("Imagem ignorada antes da chamada ao LLM.")
                    self._save_rejected_image(
                        image_path, profile_name, self._motivo_ignorada
                    )
                    skipped_by_ocr += 1
                else:
                    self._extract_data(image_path, profile_name, analysis)
                    successes += 1
            except Exception:
                failures += 1
                logger.exception(
                    "FALHA ao processar a imagem %s", image_path
                )

            durations.append(time.monotonic() - start_time)
            average_time = sum(durations) / len(durations)
            remaining_items = len(images) - index
            logger.info(
                "Tempo desta imagem: %.1fs | média: %.1fs | "
                "restantes: %d | ETA ≈ %.1f min",
                durations[-1], average_time, remaining_items, average_time * remaining_items / 60,
            )

        total_count = time.monotonic() - batch_start_time
        logger.info("=" * 60)
        logger.info(
            "RESUMO: %d sucessos | %d ignoradas pelo OCR | %d falhas | %d ofertas | "
            "%d produtos (%d compartilhadas) | %d avisos | %.1f min",
            successes, skipped_by_ocr, failures,
            self._stats["ofertas"], self._stats["produtos"],
            self._stats["compartilhadas"], self._stats["avisos"],
            total_count / 60,
        )

    # ------------------------------------------------------------------
    # Imagem: recortes e preparação
    # ------------------------------------------------------------------

    @staticmethod
    def _stack_crops(crops: list[Image.Image]) -> Image.Image:
        """
        Empilha recortes na vertical, separados por uma linha cinza

        Args:
            crops: Lista de recortes

        Returns:
            Image.Image: Imagem única com os recortes empilhados
        """
        return OllamaService.stack_crops(crops)

    @staticmethod
    def _dense_slices(image: Image.Image) -> list[Image.Image]:
        """Divide um encarte denso em faixas horizontais com sobreposição."""
        slice_height = (image.height + 1) // 2
        overlap = max(24, round(image.height * 0.08))
        middle = image.height - slice_height
        upper_end = min(image.height, middle + overlap)
        lower_start = max(0, middle - overlap)
        return [
            image.crop((0, 0, image.width, upper_end)),
            image.crop((0, lower_start, image.width, image.height)),
        ]

    def _build_crops(
        self, image: Image.Image, regions: list[RegiaoOferta]
    ) -> list[Image.Image]:
        """
        Recorta cada região e agrupa em imagens, respeitando o limite de altura

        Args:
            image: Imagem original
            regions: Regiões a recortar

        Returns:
            list[Image.Image]: Imagens prontas para enviar ao modelo
        """
        return self._ollama_service.make_crops(image, regions)

    @staticmethod
    def _prepare_image(image: Image.Image) -> tuple[bytes, dict]:
        """
        Ajusta o tamanho da imagem e gera o PNG para envio ao modelo

        Args:
            image: Imagem original

        Returns:
            tuple[bytes, dict]: Bytes do PNG e metadados do envio
        """
        return OllamaService.prepare_image(image)

    # ------------------------------------------------------------------
    # Extração
    # ------------------------------------------------------------------

    def _extract_data(
        self, image_path: Path, profile_name: str, analysis: AnaliseOCR
    ) -> None:
        """
        Envia a imagem (inteira ou recortes) ao LLM e grava o JSON final

        Além da extração principal, faz uma segunda chamada ao LLM para
        casar os produtos do product.yaml com o que aparece na imagem,
        preenchendo `produtos_encontrados` no JSON

        Args:
            image_path: Caminho da imagem
            profile_name: Nome do perfil (pasta pai da imagem)
            analysis: Resultado da análise OCR da imagem

        Returns:
            None
        """
        with Image.open(image_path) as arquivo:
            original_name = arquivo.convert("RGB")

        if analysis.use_full_image or not analysis.regions:
            images = [original_name]
            logger.info(
                "Enviando o encarte inteiro ao modelo (%s).",
                analysis.reason or "sem regiões",
            )
        else:
            images = self._build_crops(original_name, analysis.regions)
            total_area = original_name.width * original_name.height
            crop_area = sum(
                (r.box[2] - r.box[0]) * (r.box[3] - r.box[1])
                for r in analysis.regions
            )
            logger.info(
                "Recortes: %d região(ões) em %d imagem(ns) para o modelo "
                "(%.0f%% da área do encarte).",
                len(analysis.regions), len(images),
                100 * crop_area / total_area,
            )

        results: list[ResultadoExtracao] = []
        dense = analysis.dense and len(images) == 1
        if dense:
            logger.info(
                "Encarte denso: usando imagem inteira com contexto ampliado "
                "(num_ctx=%d, num_predict=%d).",
                DENSE_NUM_CTX,
                DENSE_NUM_PREDICT,
            )

        for number, image in enumerate(images, start=1):
            try:
                results.append(
                    self._query_model(
                        image, image_path, number, len(images), dense=dense
                    )
                )
            except RespostaCortada:
                fallback_regions = analysis.fallback_regions or analysis.regions
                if len(images) == 1 and fallback_regions:
                    fallback_images = self._build_crops(
                        original_name, fallback_regions
                    )
                    logger.warning(
                        "Resposta cortada. Dividindo o encarte em %d lote(s) "
                        "com regiões OCR.",
                        len(fallback_images),
                    )
                    for crop_number, crop in enumerate(
                        fallback_images, start=1
                    ):
                        results.append(
                            self._query_model(
                                crop,
                                image_path,
                                crop_number,
                                len(fallback_images),
                                attempt=2,
                            )
                        )
                    continue

                logger.warning(
                    "Resposta cortada. Tentando novamente com contexto e "
                    "saída ampliados."
                )
                try:
                    results.append(
                        self._query_model(
                            image,
                            image_path,
                            number,
                            len(images),
                            attempt=2,
                            dense=dense,
                        )
                    )
                except RespostaCortada:
                    if len(images) == 1:
                        midpoint = original_name.height // 2
                        fallback_images = [
                            original_name.crop((0, 0, original_name.width, midpoint)),
                            original_name.crop(
                                (
                                    0,
                                    midpoint,
                                    original_name.width,
                                    original_name.height,
                                )
                            ),
                        ]
                        logger.warning(
                            "Resposta cortada novamente. Dividindo o encarte "
                            "horizontalmente em %d partes.",
                            len(fallback_images),
                        )
                        results.extend(
                            self._query_model(
                                crop,
                                image_path,
                                crop_number,
                                len(fallback_images),
                                attempt=3,
                                dense=dense,
                            )
                            for crop_number, crop in enumerate(
                                fallback_images, start=1
                            )
                        )
                    else:
                        raise

        # encarte_valido=true sem nenhuma oferta é contradição:
        # trata como não encarte
        valid_flyers = [r for r in results if r.encarte_valido and r.ofertas]
        if not valid_flyers:
            self._save_rejected_image(
                image_path, profile_name, "modelo_nao_confirmou_encarte",
                analysis.found_products,
            )
            logger.info("Não é encarte. Pulando para a próxima.")
            return

        result = ResultadoExtracao(
            encarte_valido=True,
            validade=next((r.validade for r in valid_flyers if r.validade), None),
            ofertas=[offer for r in valid_flyers for offer in r.ofertas],
        )

        logger.info(
            "Encarte confirmado: %d ofertas | validade: %s",
            len(result.ofertas),
            result.validade or "não informada",
        )

        lines = self._expand_offers(result)
        if not lines:
            # o modelo disse "encarte", mas nenhuma oferta com produto e
            # preço sobreviveu à validação
            self._save_rejected_image(
                image_path, profile_name, "nenhuma_oferta_valida",
                analysis.found_products,
            )
            logger.info("Nenhuma oferta válida. Tratado como não encarte.")
            return

        # Revisita com o LLM: quais produtos do YAML aparecem nesta imagem
        found_products = self._revisit_segment_products(original_name, image_path)

        self._save_json(
            image_path, profile_name, result, lines, found_products,
        )

    def _query_model(
        self,
        image: Image.Image,
        image_path: Path,
        number: int,
        total_count: int,
        attempt: int = 1,
        dense: bool = False,
    ) -> ResultadoExtracao:
        """Encaminha uma imagem ao serviço Ollama e retorna sua resposta validada.

        Args:
            image: Imagem inteira ou recorte para extração.
            image_path: Caminho da imagem original.
            number: Número da imagem/recorte no lote.
            total_count: Total de imagens/recortes no lote.
            attempt: Número da tentativa para ajustar amostragem em retry.

        Returns:
            Resultado da extração validado por Pydantic.
        """
        return self._ollama_service.extract(
            image,
            image_path,
            number,
            total_count,
            attempt,
            dense=dense,
        )

    def _revisit_segment_products(
        self, image: Image.Image, image_path: Path
    ) -> list[str]:
        """Executa a segunda consulta de confirmação dos produtos do segmento.

        Args:
            image: Imagem inteira do encarte.
            image_path: Caminho da imagem original.

        Returns:
            Produtos do YAML confirmados na imagem ou lista vazia em caso de
            falha na chamada ou na validação.
        """
        return self._ollama_service.revisit_products(image, image_path)

    # ------------------------------------------------------------------
    # Expansão
    # ------------------------------------------------------------------

    def _expand_offers(
        self, result: ResultadoExtracao
    ) -> list[dict]:
        """
        Expande cada oferta em uma linha por variante de produto

        Valida preços e coerência de condição, e remove duplicatas

        Args:
            result: Resultado da extração do modelo

        Returns:
            list[dict]: Linhas de produto prontas para o JSON final
        """
        return self._silver_writer.expand_offers(result, self._stats)

    # ------------------------------------------------------------------
    # Saída em disco
    # ------------------------------------------------------------------

    def _save_debug_response(self, image_path: Path, response: str, number: int) -> None:
        """
        Salva a resposta bruta do modelo em um arquivo de debug

        Args:
            image_path: Caminho da imagem
            response: Resposta bruta do modelo
            number: Número da imagem no lote

        Returns:
            None
        """
        self._silver_writer.save_debug_response(image_path, response, number)

    def _save_debug_image(
        self, image_path: Path, image: Image.Image, number: int
    ) -> None:
        """
        Salva exatamente a imagem enviada ao modelo (para debug)

        Args:
            image_path: Caminho da imagem
            image: Imagem enviada ao modelo
            number: Número da imagem no lote

        Returns:
            None
        """
        self._silver_writer.save_debug_image(image_path, image, number)

    def _save_json(
        self, image_path: Path, profile_name: str,
        result: ResultadoExtracao, lines: list[dict],
        found_products: list[str],
    ) -> Path:
        """
        Grava o JSON final da extração (escrita atômica via .tmp)

        Args:
            image_path: Caminho da imagem
            profile_name: Nome do perfil
            result: Resultado validado da extração
            lines: Linhas de produto já expandidas
            found_products: Produtos do YAML confirmados pelo LLM

        Returns:
            Path: Caminho do JSON gravado
        """
        return self._silver_writer.save_result(
            image_path, profile_name, result, lines, found_products
        )

    def _save_rejected_image(
        self,
        image_path: Path,
        profile_name: str,
        reason: str,
        found_products: list[str] | None = None,
    ) -> Path:
        """
        Grava o JSON de uma imagem que não é (ou não interessa como) encarte

        O JSON fica com encarte_valido=false, para que a imagem não seja
        reprocessada nas próximas execuções (skip_existing) e para registrar
        o motivo

        Args:
            image_path: Caminho da imagem
            profile_name: Nome do perfil
            reason: Motivo pelo qual a imagem foi descartada
            found_products: Produtos do segmento que o OCR encontrou

        Returns:
            Path: Caminho do JSON gravado
        """
        return self._silver_writer.save_rejected(
            image_path, profile_name, reason, found_products
        )

    # ------------------------------------------------------------------
    # Encerramento
    # ------------------------------------------------------------------

    def close(self) -> None:
        """
        Libera os recursos da automação (Playwright, Ollama, processo)

        Returns:
            None
        """
        if self._fechando or self._fechado:
            return
        self._fechando = True
        try:
            self._ollama_service.close(TERMINATE_EXISTING_SERVER)
            self._fechado = True
        finally:
            self._fechando = False

    def __enter__(self):
        """
        Inicializa o extrator ao entrar em um bloco `with`

        Returns:
            LLMExtractor: Instância pronta para uso
        """
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Fecha os recursos ao sair de um bloco `with`

        Args:
            exc_type: Tipo da exceção ocorrida no bloco, se houver
            exc_value: Instância da exceção ocorrida, se houver
            traceback: Rastreamento da exceção, se houver

        Returns:
            bool: False (propaga a exceção, se houver)
        """
        self.close()
        return False