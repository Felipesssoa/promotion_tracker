"""Prepara caminhos únicos para armazenar imagens na camada Bronze.

Fluxo:
    1. ``ImageStorage`` determina a raiz do projeto e a pasta Bronze.
    2. ``save_image`` organiza o destino por data e perfil/rede.
    3. O primeiro nome ``story_NNN.png`` ainda não utilizado é retornado.

Este módulo não escreve o conteúdo da imagem. O chamador recebe o caminho
livre e é responsável por gravar os bytes nesse destino.
"""

import datetime
from pathlib import Path
import re

from ..core.paths import PROJECT_PATHS


class ImageStorage:
    """Resolve destinos de arquivo para imagens coletadas.

    Os destinos seguem ``data/bronze/AAAA/MM/DD/<rede>/story_NNN.png``. A
    numeração começa em ``001`` e avança enquanto o caminho já existir.

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
    def normalize_filename(filename):
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

    def save_image(self, profile_name):
        """Retorna o próximo caminho disponível para uma imagem da rede.

        O diretório diário é criado se ainda não existir. O método procura a
        primeira numeração livre para evitar sobrescrever um arquivo existente,
        mas não grava o conteúdo da imagem no caminho retornado.

        Args:
            profile_name: Nome ou identificador do perfil/rede usado na pasta destino.

        Returns:
            Caminho ainda livre no formato ``story_NNN.png``. O chamador deve
            gravar os bytes da imagem nesse caminho.

        Raises:
            OSError: Se o diretório de destino não puder ser criado ou
                acessado.
        """
        current_datetime = datetime.datetime.now()

        year = current_datetime.strftime("%Y")
        month = current_datetime.strftime("%m")
        day = current_datetime.strftime("%d")

        filename = self.normalize_filename(profile_name)

        output_dir = self.bronze_dir / year / month / day / self.normalize_filename(filename)
        output_dir.mkdir(parents=True, exist_ok=True)

        sequence_number = 1

        while True:
            output_path = output_dir / f"story_{sequence_number:03d}.png"

            if not output_path.exists():
                break
            sequence_number += 1

        return output_path