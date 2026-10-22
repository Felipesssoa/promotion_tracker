"""Configurações compartilhadas do fluxo de extração.

Este módulo concentra os limites de desempenho, as opções de casamento do
OCR, a geometria dos recortes e as opções de ciclo de vida do Ollama. Os
valores são constantes de módulo para que o extrator e seus componentes usem
uma única configuração, sem alterar o comportamento padrão do pipeline.
"""

from __future__ import annotations

import os


def _detect_physical_cores() -> int:
    """Detecta uma quantidade conservadora de threads para inferência em CPU.

    Prefere a contagem de núcleos físicos fornecida pelo ``psutil``. Quando o
    pacote não está disponível ou falha, usa metade dos processadores lógicos,
    com mínimo de uma thread.

    Returns:
        Quantidade de threads a usar pelo ``llama.cpp``.
    """
    try:
        import psutil  # type: ignore

        physical_core_count = psutil.cpu_count(logical=False)
        if physical_core_count and physical_core_count > 0:
            return physical_core_count
    except Exception:
        pass

    logical_core_count = os.cpu_count() or 2
    return max(1, logical_core_count // 2)


# Parâmetros de inferência local pelo Ollama.
NUM_CTX = 8192
NUM_PREDICT = 4096
DENSE_NUM_CTX = 32768
DENSE_NUM_PREDICT = 16384
DENSE_NUM_GPU = -1
DENSE_OCR_PRICE_THRESHOLD = 12
MAX_LADO = 1024
MIN_LADO_ENVIO = 672
NUM_THREAD = _detect_physical_cores()
NUM_BATCH = 512

# Correspondência tolerante a erros de leitura óptica de caracteres (OCR).
MATCH_JOINED_TEXT = True
MATCH_FUZZY = True
FUZZY_THRESHOLD = 90
MIN_TOLERANT_CHARS = 5

# O valor padrão desativa o filtro para não descartar encartes por falha do OCR.
OCR_MIN_PRICES = 0

# Segunda leitura em faixas ampliadas, usada quando o OCR inicial não encontra
# nenhum produto do segmento.
OCR_USE_STRIPS = True
OCR_STRIP_HEIGHT = 960
OCR_STRIP_OVERLAP = 0.25
OCR_STRIP_SCALE = 1.5

# Geometria usada para localizar e combinar recortes de ofertas.
CROP_PRODUCTS = False
CROP_X_PADDING = 0.02
CROP_Y_PADDING = 0.012
CROP_ALIGNMENT_TOLERANCE = 0.07
CROP_MAX_PRICE_DISTANCE = 0.30
CROP_MAX_HEIGHT_AFTER_PRICE = 0.35
CROP_FIRST_LINE_GAP = 3.5
CROP_NEXT_LINE_GAP = 1.3
CROP_BELOW_PRICE_GAP = 1.0
CROP_HEIGHT_RATIO = 1.5
CROP_MIN_MERGE_OVERLAP = 0.15
CROP_FALLBACK_HEIGHT = 0.155
CROP_FALLBACK_WIDTH = 0.15
COMPOSITE_MAX_HEIGHT = MAX_LADO
COMPOSITE_GAP = 24

# Gerenciamento do processo do servidor Ollama.
MANAGE_OLLAMA_SERVER = True
TERMINATE_EXISTING_SERVER = True
OLLAMA_START_TIMEOUT = 25