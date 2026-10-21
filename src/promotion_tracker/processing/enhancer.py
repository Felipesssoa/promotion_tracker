from pathlib import Path
import cv2
import numpy as np
import os
import shutil
import tempfile

from ..core.paths import PROJECT_PATHS


class OCRImageEnhancer:
    """Pipeline de upscale limpo mantendo as cores originais."""

    def __init__(self):
        self.base_dir = PROJECT_PATHS.root
        self.bronze_dir = PROJECT_PATHS.bronze_dir
        self.models_dir = PROJECT_PATHS.models_dir

        self.model_name = "espcn"
        self.scale = 4
        self.model_path = self.models_dir / f"ESPCN_x{self.scale}.pb"

        if not self.model_path.exists():
            raise FileNotFoundError(f"❌ Modelo ESPCN_x4.pb não encontrado.")

        temporary_directory = Path(tempfile.gettempdir())
        self.safe_model_path = temporary_directory / self.model_path.name
        shutil.copy2(self.model_path, self.safe_model_path)

        self.sr = cv2.dnn_superres.DnnSuperResImpl_create()
        self.sr.readModel(str(self.safe_model_path))
        self.sr.setModel(self.model_name, self.scale)

        if self.safe_model_path.exists():
            self.safe_model_path.unlink()

        print(f"✅ IA configurada (ESPCN 4x limpo).")

    @staticmethod
    def _read_image_unicode(path: Path):
        try:
            stream = open(str(path), "rb")
            image_bytes = bytearray(stream.read())
            stream.close()
            image_array = np.asarray(image_bytes, dtype=np.uint8)
            return cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        except Exception:
            return None

    @staticmethod
    def _write_image_unicode(path: Path, image):
        try:
            extension = os.path.splitext(str(path))[1]
            success, im_buf_arr = cv2.imencode(extension, image)
            if success:
                im_buf_arr.tofile(str(path))
            return success
        except Exception:
            return False

    def process_bronze_folder(self):
        all_images = list(self.bronze_dir.rglob("*.png"))
        images = [
            image for image in all_images
            if "_ocr" not in image.stem and "_tmp" not in image.stem and "_debug" not in image.stem
        ]

        total_count = len(images)
        if total_count == 0:
            return

        print(f"📸 Processando {total_count} imagens...\n")
        for index, image_path in enumerate(images, start=1):
            print(f"[{index}/{total_count}] Processando: {image_path.name}")
            try:
                self._enhance_for_ocr(image_path)
            except Exception as e:
                print(f"   ⚠️ Erro: {e}")

    def _enhance_for_ocr(self, path: Path):
        image = self._read_image_unicode(path)
        if image is None:
            return

        # 1. Proteção de RAM (limita a 800px antes do upscale)
        height, width = image.shape[:2]
        if max(height, width) > 800:
            scale_factor = 800 / max(height, width)
            image = cv2.resize(image, (int(width * scale_factor), int(height * scale_factor)), interpolation=cv2.INTER_AREA)

        # 2. Upscale ESPCN (4x)
        # Mantém a imagem colorida e sem filtros agressivos.
        upscaled_image = self.sr.upsample(image)

        # 3. Salva a imagem colorida e nítida (sem cinza, sem binarização)
        final_path = path.with_name(f"{path.stem}_ocr.png")

        if self._write_image_unicode(final_path, upscaled_image):
            print(f"   ✔️ Imagem 4x colorida salva.")