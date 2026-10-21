from extraction.llm_extractor import LLMExtractor
from ingestion.instagram_client import InstagramClient
from consolidation.consolidator import Consolidator
def main():
    instagram = InstagramClient()

    try:
        # ---------------------------------------------------------
        # ETAPA 1: Coleta de Dados Brutos (Playwright)
        # ---------------------------------------------------------
        #page = instagram.connect()
        print("\n--- 1. Iniciando coleta de Stories ---")
        #stories = instagram.get_stories()
        print("--- Coleta finalizada ---\n")

        # ---------------------------------------------------------
        # ETAPA 2: Melhoria de Imagem (ESPCN 4x + Lanczos 2x)
        # ---------------------------------------------------------
        print("--- 2. Iniciando melhoria de imagens (Enhancer) ---")
        #try:
        #    enhancer = OCRImageEnhancer()
        #    enhancer.process_bronze_folder()
        #except Exception as e:
        #    print(f"\n[ERRO] Falha no Enhancer: {e}")
#
        # ---------------------------------------------------------
        # ETAPA 3: Extração via IA de Visão (Ollama)
        # ---------------------------------------------------------
        #print("\n--- 3. Iniciando Extração de Dados (Ollama VLM) ---")
        #try:
        #    # Você pode mudar o modelo aqui se preferir outro (ex: "llama3.2-vision")
        #    extractor = LLMExtractor(model_name="qwen2.5vl:latest")
        #    extractor.process_bronze_folder()
        #except Exception as e:
        #    print(f"\n[ERRO] Falha na extração com Ollama. O Ollama está rodando? Erro: {e}")
#
        #input("\nPressione ENTER para fechar o navegador e encerrar o programa...")
#
        # ---------------------------------------------------------
        # ETAPA 4: Consolidação de Dados (Silver -> Gold)
        # ---------------------------------------------------------
        print("\n--- 4. Iniciando Consolidação de Dados (Silver -> Gold) ---")
        try:
            consolidator = Consolidator()
            consolidator.consolidate()
        except Exception as e:
            print(f"\n[ERRO] Falha na consolidação de dados: {e}")
    finally:
        instagram.close()

if __name__ == "__main__":
    main()