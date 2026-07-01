import base64
from pathlib import Path
from io import BytesIO

import pypdfium2 as pdfium
from PIL import Image

from utils.logger import logger
from configs.config import (
    SUPPORTED_IMAGE_EXTENSIONS,
    SUPPORTED_PDF_EXTENSIONS,
)


class OcrBaseUtils:
    """Утилиты для отправки запросов через OpenAI"""
    IMAGE_EXTENSIONS = SUPPORTED_IMAGE_EXTENSIONS
    PDF_EXTENSIONS = SUPPORTED_PDF_EXTENSIONS
    SUPPORTED_EXTENSIONS = IMAGE_EXTENSIONS + PDF_EXTENSIONS

    @staticmethod
    def resize_pil_image(
        pil_image: Image.Image,
        max_dimension: int,
    ) -> Image.Image:
        """Изменяет размер изображения, сохраняя пропорции"""
        w, h = pil_image.size
        if max(w, h) > max_dimension:
            ratio = max_dimension / max(w, h)
            new_size = (int(w * ratio), int(h * ratio))
            pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
        return pil_image

    @staticmethod
    def image_to_base64(image: Image.Image | str | Path) -> str:
        """Конвертирует изображение в base64 строку"""
        if isinstance(image, Image.Image):
            buffered = BytesIO()
            image.save(buffered, format='PNG')
            image_bytes = buffered.getvalue()
        else:
            image_bytes = Path(image).read_bytes()
        image_base64 = base64.b64encode(image_bytes).decode()
        image_base64_url = f'data:image/png;base64,{image_base64}'
        return image_base64_url

    @classmethod
    def pdf_or_image_to_pil_images(cls, file_path: str | Path) -> list[Image.Image]:
        """Конвертирует PDF или изображение в список объектов PIL"""
        pil_images = []
        file_path = Path(file_path)
        if file_path.suffix in cls.PDF_EXTENSIONS:
            with pdfium.PdfDocument(file_path, password=None) as pdf:
                for i, page in enumerate(pdf):
                    bitmap = page.render(scale=1, rev_byteorder=True)
                    pil_image = bitmap.to_pil()
                    pil_images.append(pil_image)
        elif file_path.suffix in cls.IMAGE_EXTENSIONS:
            pil_image = Image.open(file_path)
            pil_images.append(pil_image)
        else:
            msg = (
                f'Неподдерживаемый формат файла: {file_path.name}'
                f'\nДолжен быть один из: {cls.SUPPORTED_EXTENSIONS}'
            )
            logger.error(msg)
        return pil_images

    @classmethod
    def build_resized_pdf(cls, pil_images: list[Image.Image], max_dimension: int | None) -> bytes:
        """Создает новый PDF файл из картинок PIL с дополнительным ресайзом"""
        new_pdf = pdfium.PdfDocument.new()
        for pil_image in pil_images:
            if max_dimension:
                pil_image = cls.resize_pil_image(pil_image, max_dimension)
            buf = BytesIO()
            pil_image.convert('RGB').save(buf, format='JPEG')
            buf.seek(0)
            image_obj = pdfium.PdfImage.new(new_pdf)
            image_obj.load_jpeg(buf)
            width, height = image_obj.get_px_size()
            matrix = pdfium.PdfMatrix().scale(width, height)
            image_obj.set_matrix(matrix)
            page = new_pdf.new_page(width, height)
            page.insert_obj(image_obj)
            page.gen_content()
        out_buf = BytesIO()
        new_pdf.save(out_buf, version=17)
        return out_buf.getvalue()
