"""Prepara caminhos únicos para armazenar imagens na camada Bronze.

Fluxo:
    1. ``ImageStorage`` determina a raiz do projeto e a pasta Bronze.
    2. ``save_image`` organiza o destino por data e perfil/rede.
    3. O nome inclui perfil e horário da captura para facilitar sua identificação.

Este módulo não escreve o conteúdo da imagem. O chamador recebe o caminho
livre e é responsável por gravar os bytes nesse destino.
"""

import datetime
from pathlib import Path
import re

from ..common.paths import PROJECT_PATHS


class ImageStorage:
    """Resolve destinos de arquivo para imagens coletadas.

    Os destinos seguem
    ``data/bronze/AAAA/MM/DD/<perfil>/<perfil>_HH_MM_SS.png``. Se já existir
    uma captura do mesmo perfil no mesmo segundo, um sufixo numérico evita
    sobrescrever o arquivo anterior.

    Attributes:
        base_dir: Raiz do repositório, calculada a partir deste módulo.
        bronze_dir: Diretório usado como raiz do armazenamento Bronze.
    """

    def __init__(self):
        """Inicializa as pastas-base sem criar diretórios no disco.

        Returns:
            ``None``.
        """
        self.base_dir = PROJECT_PATHS.root
        self.bronze_dir = PROJECT_PATHS.bronze_dir

    @staticmethod
    def normalize_filename(filename: str) -> str:
        """Sanitiza o nome da rede para uso em um componente de caminho.

        Remove espaços externos e ``@`` iniciais e substitui caracteres
        inválidos em nomes de arquivo no Windows por sublinhados. Valores
        vazios após a limpeza recebem o nome ``rede_desconhecida``.

        Args:
            filename: Nome ou identificador da rede/perfil.

        Returns:
            Nome seguro para uso como pasta. A extensão não é adicionada.
        """
        safe_name = str(filename).strip().lstrip("@")
        safe_name = re.sub(r'[<>:"/\\|?*]', "_", safe_name)
        return safe_name or "rede_desconhecida"

    def save_image(self, profile_name: str) -> Path:
        """Retorna o próximo caminho disponível para uma imagem do perfil.

        O diretório diário é criado se ainda não existir. O nome inclui o
        horário local em que o caminho é gerado, com precisão de segundos.
        Capturas do mesmo perfil e segundo recebem um sufixo numérico para
        evitar sobrescrever arquivos existentes. O método não grava a imagem.

        Args:
            profile_name: Nome ou identificador do perfil/rede usado na pasta destino.

        Returns:
            Caminho ainda livre no formato ``<perfil>_HH_MM_SS.png``. O
            chamador deve gravar os bytes da imagem nesse caminho.

        Raises:
            OSError: Se o diretório de destino não puder ser criado ou
                acessado.
        """
        current_datetime = datetime.datetime.now()

        year = current_datetime.strftime("%Y")
        month = current_datetime.strftime("%m")
        day = current_datetime.strftime("%d")

        filename = self.normalize_filename(profile_name)

        output_dir = self.bronze_dir / year / month / day / filename
        output_dir.mkdir(parents=True, exist_ok=True)

        time_suffix = current_datetime.strftime("%H_%M_%S")
        output_path = output_dir / f"{filename}_{time_suffix}.png"
        sequence_number = 2

        while output_path.exists():
            output_path = output_dir / (
                f"{filename}_{time_suffix}_{sequence_number:03d}.png"
            )
            sequence_number += 1

        return output_path