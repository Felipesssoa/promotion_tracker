
import logging
import socket
import subprocess
import time
from pathlib import Path

from dotenv import load_dotenv
from playwright.sync_api import (
    Browser,
    BrowserContext,
    Page,
    Playwright,
    TimeoutError as PlaywrightTimeout,
    sync_playwright,
)

from ..common.paths import PROJECT_PATHS
from .story_collector import StoryCollector


logger = logging.getLogger(__name__)


class InstagramClient:
    """
    Gerencia a inicialização do Microsoft Edge e o acesso ao Instagram

    Utiliza um perfil exclusivo do navegador para manter a sessão autenticada
    e permite realizar o login manualmente quando necessário, o login sempre manual

    A conexão com o navegador é feita pelo Playwright através da porta de
    depuração remota, permitindo controlar as páginas abertas no Edge.

    Ordem de execução das funções:
        __init__() -> connect() -> _start_edge() -> _connect_edge() -> is_logged_in() |
        se logado -> ok | senao -> stop_edge() -> _open_manual_login() |

    Explicação sobre o login:
        Se não estiver logado, ele encerra a instância de automação e abre uma janela sem depuração para permitir
        a autenticação manual sem ser barrado pelo navegador. Apóss o login, ele reestabelece a conexão com o navegador
        com a porta de depuração remota habilitada e verifica se a autenticação ainda é valida
    """

    EDGE_PATHS = [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    ]

    DEBUG_PORT = 9222  # Porta utilizada para a comunicação do Playwright com o navegador
    LOGIN_URL = "https://www.instagram.com/accounts/login/"
    HOME_URL = "https://www.instagram.com/"

    PASSWORD_SELECTOR = 'input[name="password"], input[type="password"]'  # Campos que indicam a tela de login
    FEED_SELECTOR = 'main[role="main"]'  # Área principal da página do Instagram, vai ser usado pra saber se está na home e logado
    LOGGED_SELECTOR = 'a[href="/direct/inbox/"]'  # Link para as mensagens, usado como indicador de sessão autenticada

    def __init__(self) -> None:
        """
        Inicializa os atributos necessários para controlar o navegador

        Define o diretório do projeto, o local onde os dados da sessão serão
        armazenados e as referências aos objetos do Playwright

        O perfil separado evita depender do perfil pessoal do Edge e permite
        reutilizar a autenticação entre execuções

        Returns:
            None
        """
        self.base_dir = PROJECT_PATHS.root
        self.profile_dir = PROJECT_PATHS.edge_profile_dir

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
        Verifica se a porta de depuração do Edge ta aceitando conexões

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
        Localiza o executável do Microsoft Edge no computador

        Percorre os caminhos configurados na classe e retorna o primeiro
        executável encontrado

        Returns:
            str: Caminho completo do executável do Edge

        Raises:
            FileNotFoundError: Se o navegador não for encontrado nos caminhos
            configurados
        """
        for path in self.EDGE_PATHS:
            if Path(path).exists():
                return path

        raise FileNotFoundError(
            "Microsoft Edge não encontrado nos caminhos configurados"
        )

    def _connect_edge(self) -> None:
        """
        Estabelece a conexão entre o Playwright e o Edge em execução

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
                    "Não foi encontrado um contexto ativo no Edge"
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
        Inicia o Microsoft Edge com a depuração remota habilitada

        Localiza o executável, prepara o diretório do perfil e inicia o
        navegador com os argumentos necessários para permitir a conexão
        pelo Playwright

        Aguarda a disponibilização da porta de depuração por até 30 segundos
        e conecta ao navegador assim que ele estiver pronto

        Raises:
            RuntimeError: Se a porta já estiver em uso ou o Edge for encerrado
            antes de iniciar a depuração
            TimeoutError: Se a porta não ficar disponível dentro do prazo
            FileNotFoundError: Se o executável do Edge não for encontrado
        """
        if self._port_open():
            raise RuntimeError(
                f"A porta {self.DEBUG_PORT} já está em uso. "
                "Feche o navegador de automação antes de continuar"
            )

        edge_path = self._find_edge()
        self.profile_dir.mkdir(parents=True, exist_ok=True)

        args = [
            edge_path,
            f"--user-data-dir={self.profile_dir}",
            f"--remote-debugging-port={self.DEBUG_PORT}",
            "--no-first-run",
            "--no-default-browser-check",
            self.HOME_URL,
        ]

        # Popen inicia o navegador sem bloquear a execução do Python
        self.edge_process = subprocess.Popen(args)

        for _ in range(30):
            if self._port_open():
                self._connect_edge()
                return

            # Interrompe a espera se o processo do navegador for encerrado
            if self.edge_process.poll() is not None:
                raise RuntimeError(
                    "O Edge foi encerrado antes de iniciar a depuração"
                )

            time.sleep(1)

        self._stop_edge()

        raise TimeoutError(
            "O Edge não iniciou a porta de depuração no tempo esperado"
        )

    def _stop_edge(self) -> None:
        """
        Encerra a conexão do Playwright e o processo do Edge iniciado pela classe

        Limpa as referências aos objetos utilizados durante a automação
        e tenta encerrar o processo do navegador de forma controlada

        Se o processo não terminar dentro de cinco segundos, força o
        encerramento para evitar que ele permaneça ativo

        Returns:
            None
        """
        if self.playwright:
            try:
                self.playwright.stop()
            except Exception:
                pass

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
        Abre o Edge sem depuração para permitir a autenticação manual

        Utiliza o mesmo diretório de perfil da automação para que os dados
        de sessão possam ser reutilizados na próxima conexão

        O método aguarda o encerramento do processo iniciado antes de
        continuar o fluxo de autenticação

        Returns:
            None
        """
        edge_path = self._find_edge()
        self.profile_dir.mkdir(parents=True, exist_ok=True)

        args = [
            edge_path,
            f"--user-data-dir={self.profile_dir}",
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

    def _is_logged_in(self, timeout: int = 10) -> bool:
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
            self.page.goto(
                self.HOME_URL,
                wait_until="domcontentloaded",
            )
            self.page.wait_for_timeout(2000)

            self.page.locator(self.LOGGED_SELECTOR).first.wait_for(
                state="visible",
                timeout=timeout * 1000,
            )
            return True

        except PlaywrightTimeout:
            pass

        try:
            self.page.locator(self.PASSWORD_SELECTOR).first.wait_for(
                state="visible",
                timeout=2000,
            )
            return False

        except PlaywrightTimeout:
            pass

        try:
            self.page.locator(self.FEED_SELECTOR).first.wait_for(
                state="visible",
                timeout=3000,
            )
            return True

        except PlaywrightTimeout:
            return False

    def connect(self) -> Page:
        """
        Inicia o navegador e garante que o Instagram esteja autenticado

        Primeiro abre o Edge e verifica se a sessão existente ainda é válida
        Caso não esteja autenticada, encerra a instância de automação, abre
        o navegador para login manual e tenta estabelecer uma nova conexão

        Returns:
            Page: Página do Instagram pronta para ser utilizada pela automação

        Raises:
            RuntimeError: Se a porta continuar ocupada ou se não for possível
            confirmar a autenticação após o login manual
            FileNotFoundError: Se o executável do Edge não for encontrado
            TimeoutError: Se o navegador não iniciar dentro do prazo
        """
        self._start_edge()

        if self._is_logged_in():
            logger.info("Sessão do Instagram encontrada.")
            return self.page

        logger.info("Sessão não autenticada. Será necessário fazer login.")

        self._stop_edge()
        self._open_manual_login()

        if self._port_open():
            raise RuntimeError(
                f"A porta {self.DEBUG_PORT} continua em uso. "
                "Feche todas as janelas do Edge e tente novamente"
            )

        self._start_edge()

        if not self._is_logged_in():
            self.close()
            raise RuntimeError(
                "Não foi possível confirmar o login no Instagram"
            )

        logger.info("Login confirmado.")
        return self.page

    def close(self) -> None:
        """
        Encerra os recursos utilizados pela automação

        Delega o encerramento do Playwright e do processo do Edge ao método
        responsável pela limpeza dos recursos

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
