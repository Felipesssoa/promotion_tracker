"""
Responsável por coordenar a coleta dos Stories do Instagram

A classe utiliza uma página Playwright já autenticada para acessar os
perfis configurados, aguardar o carregamento dos Stories e salvar
capturas das mídias encontradas

Responsabilidades:
    - Carregar os perfis monitorados a partir de networks.yaml
    - Navegar até os Stories de cada perfil
    - Aguardar o carregamento da mídia
    - Ocultar temporariamente os elementos da interface do Instagram
    - Salvar a imagem do Story no diretório de saída
    - Restaurar a interface após a captura

O navegador, a autenticação e a conexão com o Instagram são
responsabilidades do InstagramClient

Fluxo principal:
    InstagramClient
        -> StoryCollector
            -> _load_networks()
            -> get_stories()
                -> _wait_story_load()
                -> capture_story()
                    -> _prepare_story_capture()
                    -> _save_story_screenshot()
                    -> _restore_interface()
"""

from datetime import datetime
from time import sleep
from pathlib import Path
from typing import Any

import yaml
from playwright.sync_api import (
    Page,
    TimeoutError as PlaywrightTimeoutError,
)
from ..core.paths import PROJECT_PATHS
from .image_storage import ImageStorage

class StoryCollector:
    """
    Coordena a coleta e a captura dos Stories dos perfis monitorados

    Recebe uma página Playwright previamente conectada ao Instagram
    Os perfis são carregados do arquivo config/networks.yaml e as
    imagens são armazenadas no diretório output/stories

    Attributes:
        page (Page): Página do Playwright utilizada para navegar no Instagram
        base_dir (Path): Diretório raiz do projeto
        networks_path (Path): Caminho do arquivo de configuração dos perfis
        output_dir (Path): Diretório onde as capturas serão armazenadas
        profiles (list[dict[str, Any]]): Perfis carregados da configuração
    """

    # Marcadores temporários inseridos no DOM durante a captura
    MEDIA_ATTR = "data-story-media"
    HIDE_ATTR = "data-story-hide"

    # Área mínima, em pixels quadrados, para considerar uma mídia como Story
    # Ajuda a descartar avatares, ícones e imagens pequenas
    MIN_MEDIA_AREA = 40_000

    # Seletor utilizado para identificar o indicador de carregamento
    SPINNER_SELECTOR = 'svg[aria-label="Carregando..."]'

    # Script que verifica se existe uma mídia grande carregada na página
    JS_MEDIA_READY = """
        (min_area) => {
            return [...document.querySelectorAll('video, img')].some(el => {
                const rect = el.getBoundingClientRect();
                const area = rect.width * rect.height;

                const visible =
                    rect.width > 0 &&
                    rect.height > 0 &&
                    rect.bottom > 0 &&
                    rect.top < window.innerHeight;

                const loaded = el.tagName === 'VIDEO'
                    ? el.readyState >= 2
                    : el.complete && el.naturalWidth > 0;

                return visible && area >= min_area && loaded;
            });
        }
    """

    # Script que identifica a maior mídia visível e oculta os demais
    # elementos da interface sem alterar o espaço ocupado pela mídia
    JS_PREPARE_CAPTURE = """
        ({ media_attr, hide_attr, min_area }) => {
            let media = null;
            let largest_area = 0;

            for (const element of document.querySelectorAll('video, img')) {
                const rect = element.getBoundingClientRect();
                const area = rect.width * rect.height;

                const visible =
                    rect.width > 0 &&
                    rect.height > 0 &&
                    rect.bottom > 0 &&
                    rect.top < window.innerHeight;

                if (visible && area > largest_area) {
                    media = element;
                    largest_area = area;
                }
            }

            if (!media || largest_area < min_area) {
                return false;
            }

            media.setAttribute(media_attr, '1');

            let current = media;

            while (current && current !== document.body) {
                const parent = current.parentElement;

                if (!parent) {
                    break;
                }

                for (const sibling of parent.children) {
                    if (sibling !== current) {
                        sibling.setAttribute(hide_attr, '1');
                    }
                }

                current = parent;
            }

            const style = document.createElement('style');
            style.id = 'story-hide-style';
            style.textContent =
                `[${hide_attr}="1"] { visibility: hidden !important; }`;

            document.head.appendChild(style);

            if (media.tagName === 'VIDEO') {
                media.pause();
            }

            return true;
        }
    """

    # Script que remove os marcadores temporários e restaura a interface
    JS_RESTORE_INTERFACE = """
    ({ media_attr, hide_attr }) => {
        // Remove todas as regras de estilo criadas pela captura.
        document.querySelectorAll('#story-hide-style').forEach(style => {
            style.remove();
        });

        // Restaura todos os elementos marcados como ocultos.
        document.querySelectorAll(`[${hide_attr}]`).forEach(element => {
            element.removeAttribute(hide_attr);
        });

        // Remove a marcação temporária da mídia capturada.
        document.querySelectorAll(`[${media_attr}]`).forEach(element => {
            element.removeAttribute(media_attr);
        });

        // Força o navegador a recalcular o layout.
        document.documentElement.offsetHeight;

        // Retorna um diagnóstico da restauração.
        return {
            style_exists: document.querySelector('#story-hide-style') !== null,
            hidden_elements: document.querySelectorAll(`[${hide_attr}]`).length,
            media_elements: document.querySelectorAll(`[${media_attr}]`).length
        };
    }
"""

    def __init__(self, page: Page) -> None:
        """
        Inicializa o coletor e carrega a configuração dos perfis.

        Args:
            page (Page): Página Playwright já conectada ao Instagram.

        Raises:
            FileNotFoundError: Se networks.yaml não existir.
            ValueError: Se a configuração não possuir uma lista válida
                de redes e perfis.
        """
        self.page = page

        self.base_dir = PROJECT_PATHS.root
        self.networks_path = PROJECT_PATHS.networks_config
        self.output_dir = PROJECT_PATHS.bronze_dir

        self.profiles = self._load_networks()
        self.image_storage = ImageStorage() # Responsável pela gravação das imagens

    def _load_networks(self) -> list[dict[str, Any]]:
        """
        Carrega os perfis monitorados a partir do arquivo YAML

        O arquivo deve conter uma chave 'networks' com uma lista
        de perfis configurados

        Returns:
            list[dict[str, Any]]: Lista de configurações dos perfis

        Raises:
            FileNotFoundError: Se o arquivo não for encontrado
            ValueError: Se o conteúdo não tiver a estrutura esperada
        """
        if not self.networks_path.is_file():
            raise FileNotFoundError(
                f"Arquivo de redes não encontrado: {self.networks_path}"
            )

        with self.networks_path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file) or {}

        if not isinstance(config, dict):
            raise ValueError(
                "A raiz de networks.yaml deve ser um objeto YAML."
            )

        networks = config.get("networks")

        if not isinstance(networks, list):
            raise ValueError(
                "O arquivo networks.yaml deve conter uma lista chamada "
                "'networks'."
            )

        for index, profile in enumerate(networks):
            if not isinstance(profile, dict):
                raise ValueError(
                    f"O item {index} de 'networks' deve ser um objeto."
                )

            if not profile.get("profile"):
                raise ValueError(
                    f"O item {index} de 'networks' não possui o campo "
                    "'profile'."
                )

        return networks

    def get_stories(self) -> list[dict[str, Any]]:
        """
        Percorre os perfis configurados e captura seus Stories

        Para cada perfil, navega até a página de Stories, aguarda
        o carregamento e tenta salvar uma captura da mídia encontrada

        Falhas individuais são registradas no resultado para permitir
        que os demais perfis continuem sendo processados

        Returns:
            list[dict[str, Any]]: Resultado da coleta de cada perfil,
            incluindo o status e o caminho da imagem quando disponível
        """
        results: list[dict[str, Any]] = []

        self.output_dir.mkdir(parents=True, exist_ok=True)

        for profile_config in self.profiles:
            profile = str(profile_config["profile"]).strip().lstrip("@")

            if not profile:
                continue

            print(f"Acessando Stories do perfil: @{profile}")

            try:
                self.page.goto(
                    f"https://www.instagram.com/stories/{profile}/",
                    wait_until="domcontentloaded",
                )

                #espera carregar
                self._wait_story_load(profile)

                #loop para passar todos os stories do perfil
                last_story = False
                while not last_story:
                    # Captura o Story e salva na pasta bronze
                    self.capture_story(profile_name=profile)

                    results.append(
                        {
                            "profile": profile,
                            "status": "success",
                            "collected_at": datetime.now().isoformat(
                                timespec="seconds"
                            ),
                            "error": None,
                        }
                    )

                    #restaura a interface do instagram após a captura
                    self._restore_interface()

                    # clica no bota ode avançar pro prox story
                    if not self.next_story():
                        last_story = True

                    #espera um tempo pro prox loop
                    sleep(0.5)



            except Exception as exc:
                print(
                    f"Falha ao coletar Stories de @{profile}: {exc}"
                )

                results.append(
                    {
                        "profile": profile,
                        "status": "error",
                        "image_path": None,
                        "collected_at": datetime.now().isoformat(
                            timespec="seconds"
                        ),
                        "error": str(exc),
                    }
                )


        return results

    def _wait_story_load(self, profile: str) -> None:
        """
        Aguarda o carregamento do Story do perfil selecionado

        Tenta clicar no botão 'Ver story', caso esteja disponível
        Em seguida, aguarda o indicador de carregamento desaparecer.

        Args:
            profile (str): Nome do perfil que está sendo acessado

        Raises:
            PlaywrightTimeoutError: Se o indicador de carregamento
                não desaparecer dentro do prazo configurado
        """
        story_button = self.page.get_by_role(
            "button",
            name="Ver story",
        )

        try:
            story_button.wait_for(
                state="visible",
                timeout=5_000,
            )

            story_button.click()

        except PlaywrightTimeoutError:
            print(
                f"Botão 'Ver story' não apareceu para @{profile}. "
                "Continuando a verificação."
            )

        # O indicador pode aparecer e desaparecer rapidamente
        try:
            self.page.locator(self.SPINNER_SELECTOR).wait_for(
                state="visible",
                timeout=3_000,
            )
        except PlaywrightTimeoutError:
            pass

        self.page.locator(self.SPINNER_SELECTOR).wait_for(
            state="hidden",
            timeout=30_000,
        )

        print(f"Carregamento concluído para @{profile}.")

    def capture_story(self, profile_name: str) -> Path:
        """Captura a mídia do Story, salva no Bronze e restaura a interface."""

        print(f"[CAPTURA] Iniciando captura de @{profile_name}")

        try:
            self.page.wait_for_function(
                self.JS_MEDIA_READY,
                arg=self.MIN_MEDIA_AREA,
                timeout=15_000,
            )
            print("[CAPTURA] Mídia carregada")

            prepared_capture = self.page.evaluate(
                self.JS_PREPARE_CAPTURE,
                {
                    "media_attr": self.MEDIA_ATTR,
                    "hide_attr": self.HIDE_ATTR,
                    "min_area": self.MIN_MEDIA_AREA,
                },
            )

            if not prepared_capture:
                raise RuntimeError(
                    "Não foi encontrada uma mídia adequada."
                )

            print("[CAPTURA] Interface preparada")

            self.page.wait_for_timeout(300)

            average_time = self.page.locator(
                f'[{self.MEDIA_ATTR}="1"]'
            )

            output_path = self.image_storage.save_image(profile_name)

            print(f"[CAPTURA] Caminho recebido: {output_path}")

            average_time.screenshot(
                path=str(output_path),
                timeout=20_000,
            )

            output_path = Path(output_path)

            if not output_path.is_file():
                raise RuntimeError("O arquivo não foi criado.")

            if output_path.stat().st_size == 0:
                raise RuntimeError("A imagem foi criada, mas está vazia.")

            print(f"[CAPTURA] Imagem salva: {output_path.resolve()}")

            return output_path

        finally:
            self._restore_interface()

    def _restore_interface(self) -> None:
        """Restaura a interface antes de avançar para o próximo Story."""

        try:
            if self.page.is_closed():
                print("[RESTAURAÇÃO] A página está fechada.")
                return

            result = self.page.evaluate(
                self.JS_RESTORE_INTERFACE,
                {
                    "media_attr": self.MEDIA_ATTR,
                    "hide_attr": self.HIDE_ATTR,
                },
            )

            print(f"[RESTAURAÇÃO] Diagnóstico: {result}")

            if (
                result["style_exists"]
                or result["hidden_elements"] > 0
                or result["media_elements"] > 0
            ):
                raise RuntimeError(
                    "A interface continua com elementos marcados."
                )

            print("[RESTAURAÇÃO] Interface restaurada com sucesso.")

        except Exception as exc:
            print(f"[RESTAURAÇÃO] Erro ao restaurar: {exc}")
            raise

    def is_last_story(self):
        can_advance = self.page.locator('svg[aria-label="Avançar"]')

        if can_advance.count() > 0:
            False # nao é o ultimo
        else:
            True # é o ultimo

    def next_story(self):
        can_advance = self.page.locator('svg[aria-label="Avançar"]')

        try:
            can_advance.click(timeout=3000)
            return True

        except PlaywrightTimeoutError:
            print("Botão 'Avançar' não encontrado em 3 segundos. Seguindo...")
            return False