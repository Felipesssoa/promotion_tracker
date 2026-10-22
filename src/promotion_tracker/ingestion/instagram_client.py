import logging
import shutil
import socket
import subprocess
import time
from pathlib import Path
from typing import ClassVar

from dotenv import load_dotenv
from playwright._impl._errors import TargetClosedError
from playwright.sync_api import (
    Browser,
    BrowserContext,
    Page,
    Playwright,
    sync_playwright,
)
from playwright.sync_api import TimeoutError as PlaywrightTimeout

from ..common.paths import PROJECT_PATHS
from .story_collector import StoryCollector

logger = logging.getLogger(__name__)


class InstagramClient:
    """
    Gerencia a inicialização do navegador (Edge, Brave ou Firefox) e o acesso
    ao Instagram

    Utiliza um perfil exclusivo do navegador para manter a sessão autenticada
    e permite realizar o login manualmente quando necessário, o login sempre manual

    A conexão com o navegador é feita pelo Playwright através da porta de
    depuração remota (Edge e Brave, ambos Chromium), permitindo controlar as
    páginas abertas no navegador.

    Ordem de execução das funções:
        __init__() -> connect() -> _start_edge() -> _connect_edge() -> is_logged_in() |
        se logado -> ok | senao -> stop_edge() -> _open_manual_login() |

    Explicação sobre o login:
        Se não estiver logado, ele encerra a instância de automação e abre uma janela sem depuração para permitir
        a autenticação manual sem ser barrado pelo navegador. Apóss o login, ele reestabelece a conexão com o navegador
        com a porta de depuração remota habilitada e verifica se a autenticação ainda é valida

    Observação sobre os nomes:
        Os métodos _start_edge, _find_edge e _stop_edge e o atributo
        edge_process foram mantidos com esses nomes por compatibilidade, mas
        valem para qualquer navegador Chromium suportado (Edge e Brave).
    """

    SUPPORTED_BROWSERS: ClassVar[tuple[str, ...]] = (
        "edge",
        "brave",
        "firefox",
        "mozilla",
    )
    BROWSER_PATHS: ClassVar[dict[str, tuple[str, ...]]] = {
        "edge": (
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
            "/usr/bin/microsoft-edge",
            "/usr/bin/microsoft-edge-stable",
        ),
        "brave": (
            r"C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe",
            r"C:\Program Files (x86)\BraveSoftware\Brave-Browser\Application\brave.exe",
            "/usr/bin/brave-browser",
            "/opt/brave.com/brave/brave",
        ),
        "firefox": (
            r"C:\Program Files\Mozilla Firefox\firefox.exe",
            r"C:\Program Files (x86)\Mozilla Firefox\firefox.exe",
            "/usr/bin/firefox",
            "/usr/bin/firefox-esr",
        ),
    }
    # Nomes dos executáveis para procurar no PATH quando os caminhos fixos falham
    BROWSER_COMMANDS: ClassVar[dict[str, tuple[str, ...]]] = {
        "edge": ("microsoft-edge", "microsoft-edge-stable"),
        "brave": ("brave-browser", "brave"),
        "firefox": ("firefox", "firefox-esr"),
    }
    BROWSER_ALIASES: ClassVar[dict[str, str]] = {
        "mozilla": "firefox",
        "brave-browser": "brave",
    }

    DEBUG_PORT = 9222  # Porta utilizada para a comunicação do Playwright com o navegador
    LOGIN_URL = "https://www.instagram.com/accounts/login/"
    HOME_URL = "https://www.instagram.com/"

    PASSWORD_SELECTOR = 'input[name="password"], input[type="password"]'  # Campos que indicam a tela de login
    FEED_SELECTOR = 'main[role="main"]'  # Área principal da página do Instagram, vai ser usado pra saber se está na home e logado
    LOGGED_SELECTOR = 'a[href="/direct/inbox/"]'  # Link para as mensagens, usado como indicador de sessão autenticada

    def __init__(self, browser: str = "edge") -> None:
        """
        Inicializa os atributos necessários para controlar o navegador

        Define o diretório do projeto, o local onde os dados da sessão serão
        armazenados e as referências aos objetos do Playwright

        O perfil separado evita depender do perfil pessoal do navegador e
        permite reutilizar a autenticação entre execuções

        Args:
            browser (str): "edge", "brave" ou "firefox" ("mozilla" também é aceito)

        Returns:
            None
        """
        normalized_browser = browser.strip().lower()
        normalized_browser = self.BROWSER_ALIASES.get(
            normalized_browser, normalized_browser
        )
        if normalized_browser not in self.BROWSER_PATHS:
            supported = ", ".join(self.SUPPORTED_BROWSERS)
            raise ValueError(
                f"Navegador inválido: {browser!r}. Use: {supported}."
            )

        self.browser_name = normalized_browser
        self.base_dir = PROJECT_PATHS.root
        self.profile_dir = (
            self.base_dir / f"{self.browser_name}_ig_profile"
        )

        # Carrega as variaveis de ambiente do arquivo .env que fica na raiz
        load_dotenv(self.base_dir / ".env")

        self.playwright: Playwright | None = None
        self.browser: Browser | None = None
        self.context: BrowserContext | None = None
        self.page: Page | None = None
        self.edge_process: subprocess.Popen | None = None

    def get_stories(self):
        if not self.page:
            raise RuntimeError("Instagram não está conectado")

        story_collector = StoryCollector(self.page)
        return story_collector.get_stories()

    def _port_open(self) -> bool:
        """
        Verifica se a porta de depuração do navegador ta aceitando conexões

        Tenta estabelecer uma conexão TCP local com a porta configurada.
        A verificação permite identificar se o navegador já iniciou a interface
        de depuração ou se existe outra aplicação utilizando essa porta

        Returns:
            bool: True se a conexão for aceita, False caso contrário
        """
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.5)

            # connect_ex retorna zero quando a conexão é estabelecida
            return sock.connect_ex(
                ("127.0.0.1", self.DEBUG_PORT)
            ) == 0

    def _find_edge(self) -> str:
        """
        Localiza o executável do navegador escolhido no computador

        Percorre os caminhos configurados na classe e retorna o primeiro
        executável encontrado. Se nenhum existir, procura os nomes
        conhecidos do executável no PATH

        Returns:
            str: Caminho completo do executável do navegador

        Raises:
            FileNotFoundError: Se o navegador não for encontrado nos caminhos
            configurados nem no PATH
        """
        browser_name = self.browser_name
        for path in self.BROWSER_PATHS[browser_name]:
            if Path(path).exists():
                return path

        for command in self.BROWSER_COMMANDS[browser_name]:
            executable = shutil.which(command)
            if executable:
                return executable

        raise FileNotFoundError(
            f"{browser_name.capitalize()} não encontrado nos caminhos configurados"
        )

    def _connect_edge(self) -> None:
        """
        Estabelece a conexão entre o Playwright e o navegador em execução

        Inicia o Playwright e utiliza o protocolo CDP para se conectar ao
        navegador pela porta de depuração remota

        Depois da conexão, recupera o contexto existente e seleciona uma
        página aberta ou cria uma nova caso não existam páginas disponíveis

        Raises:
            RuntimeError: Se o navegador não possuir um contexto disponível
            Exception: Se ocorrer um erro durante a conexão ou a configuração
        """
        self.playwright = sync_playwright().start()

        try:
            # O CDP permite controlar uma instância existente do Chromium
            self.browser = self.playwright.chromium.connect_over_cdp(
                f"http://127.0.0.1:{self.DEBUG_PORT}"
            )

            if not self.browser.contexts:
                raise RuntimeError(
                    f"Não foi encontrado um contexto ativo no {self.browser_name.capitalize()}"
                )

            self.context = self.browser.contexts[0]

            # Reutiliza uma página aberta para evitar criar abas desnecessárias
            self.page = (
                self.context.pages[0]
                if self.context.pages
                else self.context.new_page()
            )

        except Exception:
            # Libera o Playwright caso a conexão não seja concluída
            self.playwright.stop()
            self.playwright = None
            raise

    def _start_edge(self) -> None:
        """
        Inicia o navegador Chromium (Edge ou Brave) com o Playwright.
        Para o Firefox, delega para _start_firefox

        Localiza o executável, prepara o diretório do perfil e inicia o
        navegador com os argumentos necessários para permitir a conexão
        pelo Playwright

        Aguarda a disponibilização da porta de depuração por até 30 segundos
        e conecta ao navegador assim que ele estiver pronto

        Raises:
            RuntimeError: Se a porta já estiver em uso ou o navegador for
            encerrado antes de iniciar a depuração
            TimeoutError: Se a porta não ficar disponível dentro do prazo
            FileNotFoundError: Se o executável do navegador não for encontrado
        """
        if self.browser_name == "firefox":
            self._start_firefox()
            return

        if self._port_open():
            raise RuntimeError(
                f"A porta {self.DEBUG_PORT} já está em uso. "
                f"Feche o {self.browser_name.capitalize()} de automação antes de continuar"
            )

        browser_path = self._find_edge()
        self.profile_dir.mkdir(parents=True, exist_ok=True)

        args = [
            browser_path,
            f"--user-data-dir={self.profile_dir}",
            f"--remote-debugging-port={self.DEBUG_PORT}",
            "--no-first-run",
            "--no-default-browser-check",
            self.HOME_URL,
        ]

        self.edge_process = subprocess.Popen(args)

        for _ in range(30):
            if self._port_open():
                self._connect_edge()
                return

            if self.edge_process.poll() is None:
                time.sleep(1)
                continue

            raise RuntimeError(
                f"O {self.browser_name.capitalize()} foi encerrado "
                "antes de iniciar a depuração"
            )

        self._stop_edge()
        raise TimeoutError(
            f"O {self.browser_name.capitalize()} não iniciou a porta "
            "de depuração no tempo esperado"
        )

    def _start_firefox(self) -> None:
        firefox_path = self._find_edge()
        self.profile_dir.mkdir(parents=True, exist_ok=True)
        self.playwright = sync_playwright().start()

        try:
            self.context = self.playwright.firefox.launch_persistent_context(
                user_data_dir=str(self.profile_dir),
                executable_path=firefox_path,
                headless=False,
                args=["--no-remote"],
            )
            self.page = (
                self.context.pages[0]
                if self.context.pages
                else self.context.new_page()
            )
            self.page.goto(self.HOME_URL, wait_until="domcontentloaded")
        except Exception:
            self.playwright.stop()
            self.playwright = None
            raise

    def _stop_edge(self) -> None:
        """
        Encerra a conexão do Playwright e o processo do navegador iniciado pela classe

        Limpa as referências aos objetos utilizados durante a automação
        e tenta encerrar o processo do navegador de forma controlada

        Se o processo não terminar dentro de cinco segundos, força o
        encerramento para evitar que ele permaneça ativo

        Returns:
            None
        """
        if self.playwright:
            try:
                if self.context:
                    self.context.close()
                self.playwright.stop()
            except Exception:
                logger.debug("Falha ao encerrar o navegador.", exc_info=True)

        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None

        if self.edge_process and self.edge_process.poll() is None:
            self.edge_process.terminate()

            try:
                self.edge_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.edge_process.kill()
                self.edge_process.wait()

        self.edge_process = None

    def _open_manual_login(self) -> None:
        """
        Abre o navegador sem depuração para permitir a autenticação manual

        Utiliza o mesmo diretório de perfil da automação para que os dados
        de sessão possam ser reutilizados na próxima conexão

        O método aguarda o encerramento do processo iniciado antes de
        continuar o fluxo de autenticação

        Returns:
            None
        """
        browser_path = self._find_edge()
        self.profile_dir.mkdir(parents=True, exist_ok=True)

        # Edge e Brave (Chromium) usam --user-data-dir; o Firefox usa --profile
        profile_args = (
            [f"--profile={self.profile_dir}"]
            if self.browser_name == "firefox"
            else [f"--user-data-dir={self.profile_dir}"]
        )
        args = [
            browser_path,
            *profile_args,
            "--no-first-run",
            "--no-default-browser-check",
            self.LOGIN_URL,
        ]

        process = subprocess.Popen(args)

        logger.info(
            "Faça login no Instagram na janela aberta e feche o navegador "
            "para continuar."
        )

        # Aguarda o processo iniciado pelo comando terminar
        process.wait()

    def _is_logged_in(
        self,
        timeout: int = 10,
        *,
        navigate: bool = True,
    ) -> bool:
        """
        Verifica se o Instagram está acessível com uma sessão autenticada

        Abre a página inicial e procura indicadores de sessão ativa
        Primeiro verifica o link das mensagens, depois o campo de senha e,
        por último, a área principal da página

        Args:
            timeout (int): Tempo máximo, em segundos, para aguardar o indicador
            principal de autenticação

        Returns:
            bool: True se encontrar um indicador de acesso autenticado,
            False se encontrar a tela de login ou não confirmar a sessão
        """
        if not self.page:
            return False

        try:
            if navigate:
                self.page.goto(
                    self.HOME_URL,
                    wait_until="domcontentloaded",
                )
                self.page.wait_for_timeout(2000)

            if self.page.locator(self.LOGGED_SELECTOR).count() > 0:
                return True

            if self.page.locator(self.PASSWORD_SELECTOR).count() > 0:
                return False

            return self.page.locator(self.FEED_SELECTOR).count() > 0

        except (PlaywrightTimeout, TargetClosedError):
            return False

    def _wait_for_manual_login(self, timeout: int = 900) -> None:
        if not self.page:
            raise RuntimeError(
                f"{self.browser_name.capitalize()} não está conectado"
            )

        logger.info(
            f"Faça login no Instagram na janela aberta do "
            f"{self.browser_name.capitalize()}. "
            "A coleta continuará automaticamente após a autenticação."
        )
        self.page.goto(self.LOGIN_URL, wait_until="domcontentloaded")

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._is_logged_in(timeout=2, navigate=False):
                return
            self.page.wait_for_timeout(2_000)

        raise TimeoutError(
            f"Login no Instagram não foi confirmado em {timeout} segundos"
        )

    def connect(self) -> Page:
        """
        Inicia o navegador e garante que o Instagram esteja autenticado

        Primeiro abre o navegador e verifica se a sessão existente ainda é válida
        Caso não esteja autenticada, encerra a instância de automação, abre
        o navegador para login manual e tenta estabelecer uma nova conexão

        Returns:
            Page: Página do Instagram pronta para ser utilizada pela automação

        Raises:
            RuntimeError: Se a porta continuar ocupada ou se não for possível
            confirmar a autenticação após o login manual
            FileNotFoundError: Se o executável do navegador não for encontrado
            TimeoutError: Se o navegador não iniciar dentro do prazo
        """
        self._start_edge()

        if self._is_logged_in():
            logger.info("Sessão do Instagram encontrada.")
            return self.page

        logger.info("Sessão não autenticada. Será necessário fazer login.")

        self._wait_for_manual_login()

        logger.info("Login confirmado.")
        return self.page

    def close(self) -> None:
        """
        Encerra os recursos utilizados pela automação

        Delega o encerramento do Playwright e do processo do navegador ao
        método responsável pela limpeza dos recursos

        Returns:
            None
        """
        self._stop_edge()

    def __enter__(self):
        """
        Inicializa a conexão ao entrar em um bloco with

        Returns:
            InstagramClient: Instância conectada ao Instagram
        """
        self.connect()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """
        Fecha os recursos ao sair de um bloco with

        Args:
            exc_type: Tipo da exceção ocorrida no bloco, se houver
            exc_value: Instância da exceção ocorrida, se houver
            traceback: Rastreamento da exceção, se houver

        Returns:
            None
        """
        self.close()