"""Ponto de entrada: lê os argumentos, monta as etapas e executa o pipeline.

Sem argumentos de seleção, em um terminal interativo, mostra um menu para
escolher por onde começar. Para automatizar (cron, scripts), use as flags.

Exemplos:
    python src/promotion_tracker/main.py                       # menu interativo
    python src/promotion_tracker/main.py -y                    # tudo, sem perguntas
    python src/promotion_tracker/main.py --from extract        # pula a coleta
    python src/promotion_tracker/main.py --skip collect        # idem
    python src/promotion_tracker/main.py --from normalize      # só normaliza e consolida
    python src/promotion_tracker/main.py --from extract --until extract --reprocess --max-images 5
    python src/promotion_tracker/main.py --stages extract normalize
    python src/promotion_tracker/main.py --from extract --dry-run
"""
import argparse
import dataclasses
import logging
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

logger = logging.getLogger(__name__)

BROWSERS = ("edge", "brave", "firefox", "mozilla")


def _ensure_src_on_path() -> None:
    source_root = str(Path(__file__).resolve().parents[1])
    if source_root not in sys.path:
        sys.path.insert(0, source_root)


# --------------------------------------------------------------------------- seleção
def resolve_stages(
    names: Sequence[str],
    stages: Sequence[str] | None = None,
    start: str | None = None,
    until: str | None = None,
    skip: Sequence[str] = (),
) -> list[str]:
    """Etapas a executar, sempre na ordem do pipeline."""
    wanted = set(stages) if stages else set(names)
    first = names.index(start) if start else 0
    last = names.index(until) if until else len(names) - 1
    chosen = [
        name for index, name in enumerate(names)
        if name in wanted and first <= index <= last and name not in skip
    ]
    if not chosen:
        raise ValueError("Nenhuma etapa selecionada: confira --stages/--from/--until/--skip.")
    return chosen


# --------------------------------------------------------------------------- perguntas
def _ask(prompt: str, input_fn: Callable[[str], str]) -> str:
    try:
        return input_fn(prompt).strip()
    except EOFError:  # sem terminal: assume o padrão
        return ""


def ask_choice(
    title: str,
    options: Sequence[tuple[str, str]],
    default_index: int,
    input_fn: Callable[[str], str] = input,
) -> str:
    """Menu numerado. Enter escolhe o padrão; também aceita o nome da opção."""
    print(f"\n{title}")
    for number, (_, label) in enumerate(options, start=1):
        print(f"  {number}) {label}")
    while True:
        raw = _ask(f"Escolha [1-{len(options)}] (Enter = {default_index + 1}): ", input_fn)
        if not raw:
            return options[default_index][0]
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1][0]
        for key, _ in options:
            if raw.lower() == key:
                return key
        print("  Opção inválida.")


def ask_yes_no(question: str, default: bool, input_fn: Callable[[str], str] = input) -> bool:
    suffix = "[S/n]" if default else "[s/N]"
    while True:
        raw = _ask(f"\n{question} {suffix}: ", input_fn).lower()
        if not raw:
            return default
        if raw in ("s", "sim", "y", "yes"):
            return True
        if raw in ("n", "nao", "não", "no"):
            return False
        print("  Responda com s ou n.")


def ask_optional_int(question: str, input_fn: Callable[[str], str] = input) -> int | None:
    while True:
        raw = _ask(f"\n{question}: ", input_fn)
        if not raw:
            return None
        if raw.isdigit() and int(raw) > 0:
            return int(raw)
        print("  Digite um número inteiro positivo ou Enter para todas.")


def prompt_for_run(
    args: argparse.Namespace,
    names: Sequence[str],
    titles: dict[str, str],
    input_fn: Callable[[str], str] = input,
) -> argparse.Namespace | None:
    """Menu interativo. Devolve os args ajustados, ou None se o usuário cancelar."""
    print("\n=== Promotion Tracker ===")

    start = ask_choice(
        "Por qual etapa começar?",
        [(name, titles[name]) for name in names],
        default_index=0,
        input_fn=input_fn,
    )
    remaining = list(names[names.index(start):])
    until = ask_choice(
        "Até qual etapa executar?",
        [(name, titles[name]) for name in remaining],
        default_index=len(remaining) - 1,
        input_fn=input_fn,
    )
    chosen = resolve_stages(names, start=start, until=until)
    args.stages, args.from_stage, args.until, args.skip = chosen, None, None, []

    if "collect" in chosen:
        args.browser = ask_choice(
            "Navegador para a coleta:",
            [(browser, browser) for browser in BROWSERS],
            default_index=BROWSERS.index(args.browser),
            input_fn=input_fn,
        )
    if "extract" in chosen:
        args.reprocess = ask_yes_no(
            "Reprocessar imagens que já têm JSON na Silver (sobrescreve)?",
            default=args.reprocess,
            input_fn=input_fn,
        )
        args.max_images = ask_optional_int(
            "Limite de imagens na extração (Enter = todas)", input_fn=input_fn
        )

    print("\nPlano: " + " -> ".join(chosen))
    if not ask_yes_no("Executar?", default=True, input_fn=input_fn):
        return None
    return args


def should_prompt(args: argparse.Namespace, is_tty: bool) -> bool:
    if args.yes:
        return False
    if args.interactive:
        return True
    selection_given = bool(args.stages or args.from_stage or args.until or args.skip)
    return is_tty and not selection_given


# --------------------------------------------------------------------------- argumentos
def parse_args(defaults, stage_names: tuple[str, ...], argv: Sequence[str] | None = None) -> argparse.Namespace:
    """`defaults` é um ExtractOptions: os padrões vêm de lá, não ficam duplicados aqui."""
    parser = argparse.ArgumentParser(
        description="Pipeline de promoções",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Etapas: " + ", ".join(stage_names),
    )

    flow = parser.add_argument_group("seleção de etapas")
    flow.add_argument("--from", dest="from_stage", choices=stage_names,
                      help="Começa nesta etapa e vai até o fim (ex.: --from extract pula a coleta)")
    flow.add_argument("--until", choices=stage_names, help="Para depois desta etapa")
    flow.add_argument("--skip", nargs="+", choices=stage_names, default=[],
                      help="Pula estas etapas")
    flow.add_argument("--stages", nargs="+", choices=stage_names, default=None,
                      help="Lista exata de etapas (alternativa a --from/--until/--skip)")
    flow.add_argument("-i", "--interactive", action="store_true",
                      help="Abre o menu mesmo se houver flags de seleção")
    flow.add_argument("-y", "--yes", action="store_true",
                      help="Não pergunta nada: executa com as flags/padrões")
    flow.add_argument("--dry-run", action="store_true",
                      help="Mostra o que seria executado e sai")

    parser.add_argument(
        "--browser",
        choices=BROWSERS,
        default="edge",
        help="Navegador usado na coleta (default: edge)",
    )

    extract = parser.add_argument_group("extração")
    extract.add_argument(
        "--extractor",
        choices=("tiles", "legacy"),
        default=defaults.extractor,
        help="tiles: leitura por recortes | legacy: OCR + LLM na imagem inteira",
    )
    extract.add_argument("--backend", choices=("ollama", "anthropic"), default=defaults.backend)
    extract.add_argument("--gate", choices=("flyer", "products", "none"), default=defaults.gate,
                         help="flyer: lê todo encarte e descarta o que não for encarte | "
                              "products: só encarte com produto monitorado | none: lê tudo")
    extract.add_argument("--model", default=defaults.model,
                         help="Default: qwen2.5vl:latest (ollama) ou claude-sonnet-5-5 (API)")
    extract.add_argument("--tile-size", type=int, default=defaults.tile_size,
                         help="Tamanho do recorte em px")
    extract.add_argument("--overlap", type=float, default=defaults.overlap,
                         help="Sobreposição entre recortes (0 a 0.5)")
    extract.add_argument("--target-long-side", type=int, default=defaults.target_long_side,
                         help="Amplia recortes menores que isso (px); 0 desliga")
    extract.add_argument("--num-ctx", type=int, default=defaults.num_ctx)
    extract.add_argument("--num-predict", type=int, default=defaults.num_predict)
    extract.add_argument("--num-thread", type=int, default=defaults.num_thread,
                         help="Threads de CPU (default: núcleos físicos)")
    extract.add_argument("--num-batch", type=int, default=defaults.num_batch)
    extract.add_argument("--reprocess", action="store_true",
                         help="Sobrescreve imagens que já têm JSON na Silver (default: pula)")
    extract.add_argument("--max-images", type=int, default=defaults.max_images)
    extract.add_argument("--debug-tiles", type=Path, default=defaults.debug_tiles,
                         help="Pasta para salvar cada recorte e a resposta crua do modelo")

    args = parser.parse_args(argv)
    if args.stages and (args.from_stage or args.until or args.skip):
        parser.error("--stages não pode ser combinado com --from/--until/--skip.")
    return args


# --------------------------------------------------------------------------- execução
def main() -> int:
    _ensure_src_on_path()

    from promotion_tracker.common.logging_config import configure_logging
    from promotion_tracker.orchestration.pipeline import Pipeline
    from promotion_tracker.orchestration.stages import (
        STAGE_NAMES,
        STAGE_TITLES,
        ExtractOptions,
        build_stages,
    )

    args = parse_args(ExtractOptions(), STAGE_NAMES)

    if should_prompt(args, sys.stdin.isatty()):
        args = prompt_for_run(args, STAGE_NAMES, STAGE_TITLES)
        if args is None:
            print("Cancelado.")
            return 0

    try:
        selected = resolve_stages(
            STAGE_NAMES, args.stages, args.from_stage, args.until, args.skip
        )
    except ValueError as exc:
        print(f"Erro: {exc}", file=sys.stderr)
        return 2

    if args.dry_run:
        print("Etapas que seriam executadas: " + " -> ".join(selected))
        return 0

    configure_logging()
    logger.info("Plano: %s", " -> ".join(selected))

    option_names = {f.name for f in dataclasses.fields(ExtractOptions)}
    options = ExtractOptions(**{k: v for k, v in vars(args).items() if k in option_names})

    try:
        Pipeline(build_stages(selected, options, args.browser)).run()
    except KeyboardInterrupt:
        logger.warning("Execução interrompida pelo usuário.")
        return 130
    except Exception:
        logger.error("A execução foi interrompida por um erro.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())