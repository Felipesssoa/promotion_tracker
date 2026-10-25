"""Divide um encarte em recortes (tiles) com sobreposição.

Em vez de mandar o encarte inteiro para o modelo de visão (que redimensiona
a imagem e perde o texto pequeno), cada recorte é lido em resolução quase
nativa. A sobreposição garante que um produto cortado na borda de um tile
apareça inteiro no tile vizinho.
"""
from __future__ import annotations

import io
import math
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps


@dataclass(frozen=True)
class Tile:
    index: int
    row: int
    col: int
    box: tuple[int, int, int, int]  # x0, y0, x1, y1 na imagem original
    image: Image.Image

    def to_jpeg_bytes(self, quality: int = 92) -> bytes:
        buffer = io.BytesIO()
        self.image.save(buffer, format="JPEG", quality=quality)
        return buffer.getvalue()


def axis_cells(
    length: int,
    tile_size: int,
    overlap: float,
    tolerance: float = 1.15,
) -> list[tuple[int, int]]:
    """Divide um eixo em células com sobreposição, de tamanhos iguais.

    Se o eixo for no máximo `tolerance` vezes maior que `tile_size`, usa uma
    célula só (evita cortar uma Story de 1080 px de largura em duas).
    """
    if length <= tile_size * tolerance:
        return [(0, length)]

    ov = tile_size * overlap
    count = max(2, math.ceil((length - ov) / (tile_size - ov)))
    size = (length + (count - 1) * ov) / count
    stride = size - ov

    cells: list[tuple[int, int]] = []
    for i in range(count):
        start = round(i * stride)
        end = length if i == count - 1 else min(length, round(start + size))
        cells.append((start, end))
    return cells


def split_image(
    path: str | Path,
    tile_size: int = 1024,
    overlap: float = 0.25,
    tolerance: float = 1.15,
    target_long_side: int | None = 1280,
    max_scale: float = 2.0,
) -> tuple[list[Tile], tuple[int, int]]:
    """Retorna (tiles, (linhas, colunas)).

    `target_long_side`: tiles menores que isso são ampliados (LANCZOS) até
    `max_scale`, o que ajuda o modelo a ler letras pequenas.
    """
    with Image.open(path) as raw:
        image = ImageOps.exif_transpose(raw).convert("RGB")

    width, height = image.size
    xs = axis_cells(width, tile_size, overlap, tolerance)
    ys = axis_cells(height, tile_size, overlap, tolerance)

    tiles: list[Tile] = []
    for row, (y0, y1) in enumerate(ys):
        for col, (x0, x1) in enumerate(xs):
            crop = image.crop((x0, y0, x1, y1))
            if target_long_side:
                long_side = max(crop.size)
                if long_side < target_long_side:
                    scale = min(max_scale, target_long_side / long_side)
                    if scale > 1.01:
                        crop = crop.resize(
                            (round(crop.width * scale), round(crop.height * scale)),
                            Image.LANCZOS,
                        )
            tiles.append(Tile(len(tiles), row, col, (x0, y0, x1, y1), crop))

    return tiles, (len(ys), len(xs))
