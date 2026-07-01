# vLLM PaddleOCR docs:
# https://docs.vllm.ai/projects/recipes/en/latest/PaddlePaddle/PaddleOCR-VL.html#deploying-paddleocr-vl
# https://docs.vllm.ai/projects/recipes/en/latest/PaddlePaddle/PaddleOCR-VL.html#offline-inference-using-vllm-combined-with-pp-doclayoutv2
# https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/PaddleOCR-VL.html
# PaddleOCR docs:
# https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/PaddleOCR-VL.html#322-python-api-invocation
# https://www.paddleocr.ai/latest/en/version2.x/ppocr/blog/inference_args.html


import os
import base64
from pathlib import Path
from io import BytesIO

import pypdfium2 as pdfium
import openai
from openai import OpenAI
from tqdm import tqdm
from PIL import Image

from utils.logger import logger
from configs.config import (
    SUPPORTED_IMAGE_EXTENSIONS,
    SUPPORTED_PDF_EXTENSIONS,
    OCR_MAX_IMAGE_DIMENSION,
)


class OcrUtilsOpenai:
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
    def ocr_document(cls, client: OpenAI, file_path: str | Path) -> list[str]:
        """Отправляет документ (изображение/PDF) для OCR через OpenAI клиент"""
        import importlib
        import configs.config
        importlib.reload(configs.config)
        from configs.config import OPENAI_CHAT_COMPLETIONS_KWARGS

        TASKS = {
            'ocr': 'OCR:',
            'table': 'Table Recognition:',
            "formula": 'Formula Recognition:',
            'chart': 'Chart Recognition:',
        }
        result_ocr_texts = []
        pil_images = cls.pdf_or_image_to_pil_images(file_path)
        if not pil_images:
            return result_ocr_texts
        image_base64_urls = []
        for pil_image in pil_images:
            if OCR_MAX_IMAGE_DIMENSION:
                pil_image = cls.resize_pil_image(pil_image, max_dimension=OCR_MAX_IMAGE_DIMENSION)
            image_base64_url = cls.image_to_base64(pil_image)
            image_base64_urls.append(image_base64_url)
        logger.info(f'Кол-во изображений в документе {file_path.name}: {len(image_base64_urls)}')
        PADDLEOCRVL_MODEL_NAME = os.getenv('PADDLEOCRVL_MODEL_NAME', 'PaddleOCR-VL-1.6-0.9B')
        pbar = tqdm(total=len(image_base64_urls), desc='Детекция страниц   ', leave=False)
        for i, image_base64_url in enumerate(image_base64_urls, start=1):
            messages = [
                {
                    'role': 'user',
                    'content': [
                        {
                            'type': 'image_url',
                            'image_url': {
                                'url': image_base64_url,
                            }
                        },
                        {
                            'type': 'text',
                           'text': TASKS['ocr'],
                        }
                    ]
                }
            ]
            logger.debug('Объекты перед отправкой запроса на `client.chat.completions.create`:')
            logger.debug(f'PADDLEOCRVL_MODEL_NAME: {PADDLEOCRVL_MODEL_NAME}')
            logger.debug(f'len(image_base64_urls): {len(image_base64_urls)}')
            logger.debug(f'image_base64_urls[0][:15]: {image_base64_urls[0][:15]}')
            logger.debug(f'client.__dict__: {client.__dict__}')
            logger.debug(f'OPENAI_CHAT_COMPLETIONS_KWARGS: {OPENAI_CHAT_COMPLETIONS_KWARGS}')
            try:
                response = client.chat.completions.create(
                    model=PADDLEOCRVL_MODEL_NAME,
                    messages=messages,
                    **OPENAI_CHAT_COMPLETIONS_KWARGS,
                )
                generated_text = response.choices[0].message.content
                result_ocr_texts.append(generated_text)
                logger.info(f'Текст для страницы {i} документа {file_path} успешно распознан')
                pbar.update()
            except openai.APIConnectionError as e:
                msg = f'Ошибка соединения клиента OpenAI: {e}'
                logger.error(msg)
                return result_ocr_texts
            except Exception as e:
                msg = f'Ошибка при отправке запроса OCR страницы {i} файла {file_path.name}: {e}'
                logger.error(msg)
                pbar.update()
        return result_ocr_texts
