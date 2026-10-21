import json

from promotion_tracker.common.paths import PROJECT_PATHS
from promotion_tracker.common.products import ProductCatalog
from promotion_tracker.extraction.llm.schemas import (
    Oferta,
    ResultadoExtracao,
    Variante,
)
from promotion_tracker.extraction.ocr.analyzer import OCRAnalyzer
from promotion_tracker.extraction.silver_writer import SilverWriter
from promotion_tracker.consolidation.consolidator import Consolidator
from promotion_tracker.normalization.normalizer import Normalizer


def test_product_catalog_canonicalizes_aliases_and_uht():
    catalog = ProductCatalog()

    assert catalog.normalize_name("REQ") == "Requeijão"
    assert catalog.normalize_name("QJ") == "Queijo"
    assert catalog.normalize_name("BEB LAC") == "Bebida Láctea"
    assert catalog.normalize_name("UHT") == "Longa Vida"
    assert catalog.normalize_name("Requeija") == "Requeijão"


def test_ocr_recognizes_abbreviated_products():
    catalog = ProductCatalog()
    analyzer = OCRAnalyzer(
        catalog.products,
        product_aliases=catalog.search_terms_by_product,
    )

    found = analyzer._find_products(["REQ QJ BEB LAC UHT"])

    assert found == [
        "Longa Vida",
        "Queijo",
        "Requeijão",
        "Bebida Láctea",
    ]


def test_normalizer_writes_reviewed_json_without_changing_silver(tmp_path):
    source_dir = tmp_path / "silver"
    reviewed_dir = tmp_path / "silver_reviewed"
    source_file = source_dir / "2026" / "10" / "21" / "oferta.json"
    source_file.parent.mkdir(parents=True)
    source_record = {
        "encarte_valido": True,
        "produtos_encontrados": ["UHT"],
        "produtos": [{
            "produto": "REQ",
            "marca": "BETANIA",
            "preco_atual": 4.99,
        }],
    }
    source_file.write_text(
        json.dumps(source_record, ensure_ascii=False),
        encoding="utf-8",
    )
    catalog = ProductCatalog()
    normalizer = Normalizer(
        silver_dir=source_dir,
        reviewed_dir=reviewed_dir,
        brands_path=PROJECT_PATHS.brands_config,
        product_catalog=catalog,
    )

    generated_count = normalizer.normalize()

    reviewed_file = reviewed_dir / source_file.relative_to(source_dir)
    reviewed_record = json.loads(reviewed_file.read_text(encoding="utf-8"))
    reviewed_product = reviewed_record["produtos"][0]
    assert generated_count == 1
    assert json.loads(source_file.read_text(encoding="utf-8")) == source_record
    assert reviewed_product["produto_original"] == "REQ"
    assert reviewed_product["produto_revisado"] == "Requeijão"
    assert reviewed_product["produto"] == "Requeijão"
    assert reviewed_product["marca_original"] == "BETANIA"
    assert reviewed_product["marca_revisada"] == "Betânia"
    assert reviewed_product["marca"] == "Betânia"
    assert reviewed_record["produtos_encontrados"] == ["Longa Vida"]
    assert normalizer._review_brand("Betnia") == "Betânia"

    gold_dir = tmp_path / "gold"
    gold_file = Consolidator(
        silver_dir=reviewed_dir,
        gold_dir=gold_dir,
        products_path=PROJECT_PATHS.product_config,
    ).consolidate()
    gold_payload = json.loads(gold_file.read_text(encoding="utf-8"))
    assert gold_payload["resumo"]["linhas_produto"] == 1
    assert gold_payload["encartes"][0]["produtos"][0]["marca"] == "Betânia"

    source_file.unlink()
    assert normalizer.normalize() == 0
    assert not reviewed_file.exists()


def test_silver_writer_canonicalizes_product_alias_on_save(tmp_path):
    catalog = ProductCatalog()
    writer = SilverWriter(
        tmp_path / "bronze",
        tmp_path / "silver",
        "test-model",
        product_catalog=catalog,
    )
    result = ResultadoExtracao(
        encarte_valido=True,
        ofertas=[
            Oferta(
                variantes=[
                    Variante(
                        produto="REQ",
                        marca="BETANIA",
                        quantidade=200,
                        unidade_medida="g",
                    )
                ],
                preco_atual=4.99,
                preco_anterior=None,
                preco_condicao=None,
                condicao=None,
            )
        ],
    )

    lines = writer.expand_offers(result, {
        "avisos": 0,
        "ofertas": 0,
        "compartilhadas": 0,
        "produtos": 0,
    })

    assert lines[0]["produto"] == "Requeijão"
    assert lines[0]["produto_original"] == "REQ"
    assert lines[0]["produto_revisado"] == "Requeijão"
