"""Compara quantos produtos cada execução achou contra um gabarito manual.

gabarito.csv (cabeçalho obrigatório):
    arquivo,total
    2026/10/21/perfil/perfil_10_30_05.png,48

`arquivo` é o caminho relativo dentro de data/bronze (o mesmo valor do campo
`arquivo_origem` nos JSONs da Silver).

Uso:
    python scripts/compare_runs.py gabarito.csv data/silver data/silver_tiles
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


def load_counts(silver_dir: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for path in silver_dir.rglob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if "arquivo_origem" in data:
            counts[data["arquivo_origem"]] = int(data.get("quantidade_produtos") or 0)
    return counts


def main() -> None:
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    gabarito_path, run_dirs = Path(sys.argv[1]), [Path(p) for p in sys.argv[2:]]

    with gabarito_path.open(encoding="utf-8") as fh:
        gabarito = {r["arquivo"].strip(): int(r["total"]) for r in csv.DictReader(fh)}

    runs = {d.name: load_counts(d) for d in run_dirs}
    header = f"{'arquivo':55}{'real':>6}" + "".join(f"{name[:14]:>16}" for name in runs)
    print(header)
    print("-" * len(header))

    totals = {name: 0 for name in runs}
    total_real = 0
    for arquivo, real in gabarito.items():
        line = f"{arquivo[-54:]:55}{real:>6}"
        for name, counts in runs.items():
            found = counts.get(arquivo)
            if found is None:
                line += f"{'-':>16}"
            else:
                line += f"{found:>9} ({min(found, real) / real:>4.0%})"
                totals[name] += min(found, real)
        total_real += real
        print(line)

    print("-" * len(header))
    print("Recuperação geral (limitada ao total real): ", end="")
    print(" | ".join(f"{n}: {t / total_real:.1%}" for n, t in totals.items()))
    print("Obs.: contagem não prova acerto de nome/preço; confira uma amostra à mão.")


if __name__ == "__main__":
    main()
