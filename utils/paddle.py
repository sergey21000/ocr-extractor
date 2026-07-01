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

import requests
from paddleocr import PaddleOCRVL

from utils.logger import logger
from configs.config import (
    PADDLEOCRVL_INIT_KWARGS,
    SUPPORTED_IMAGE_EXTENSIONS,
    SUPPORTED_PDF_EXTENSIONS,
    OCR_RESULT_DIR,
)


class OcrUtilsPaddle:
    """Утилиты для отправки запросов через OpenAI"""
    IMAGE_EXTENSIONS = SUPPORTED_IMAGE_EXTENSIONS
    PDF_EXTENSIONS = SUPPORTED_PDF_EXTENSIONS
    SUPPORTED_EXTENSIONS = IMAGE_EXTENSIONS + PDF_EXTENSIONS

    @staticmethod
    def get_paddleocrvl_pipepline(vl_rec_backend: str | None = None) -> PaddleOCRVL:
        """
        Инициализирует пайплайн PaddleOCRVL
        - Если задана переменная окружения PADDLEX_CONFIG - инициализирует
            PaddleOCRVL с конфигом - подходит для запуска примера 
            с клиентом и сервером из документации
            https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/PaddleOCR-VL.html#41-method-1-deploy-using-docker-compose-recommended
        - если передан PADDLEX_CONFIG не задан и передан vl_rec_backend - 
            PaddleOCRVL подключается к серверу (переменная LLM_BASE_URL) - 
            подходит для какого-либо кастомного запуска.
            При этом все модели кроме основной 
        - если ничего из вышеперечисленного не задано - инициализируется 
            полностью локально - подходит для запуска из под контейнера paddleocr-vl
        """
        PADDLEX_CONFIG = os.getenv('PADDLEX_CONFIG')
        if PADDLEX_CONFIG:
            pipeline = PaddleOCRVL(paddlex_config=f'configs/{PADDLEX_CONFIG}')
        else:
            if not vl_rec_backend:
                pipeline = PaddleOCRVL(**PADDLEOCRVL_INIT_KWARGS)
            else:
                LLM_BASE_URL = os.getenv('LLM_BASE_URL', 'http://127.0.0.1:8080/v1')
                pipeline = PaddleOCRVL(
                    vl_rec_backend=vl_rec_backend,
                    vl_rec_server_url=LLM_BASE_URL,
                    **PADDLEOCRVL_INIT_KWARGS,
                )
        return pipeline

    @staticmethod
    def paddleocrvl_predict_and_save(
        pipeline: PaddleOCRVL,
        input_path: str | Path,
    ) -> None:
        """Запускает PaddleOCR на файле или всех файлах в папке и сохраняет результаты"""
        import importlib
        import configs.config
        importlib.reload(configs.config)
        from configs.config import PADDLEOCRVL_PREDICT_KWARGS

        output = pipeline.predict(
            input=str(input_path),
            **PADDLEOCRVL_PREDICT_KWARGS,
        )
        for res in output:
            res.print()
            ocr_result_dir = Path(OCR_RESULT_DIR) / Path(res['input_path']).stem
            res.save_to_json(save_path=ocr_result_dir)
            res.save_to_markdown(save_path=ocr_result_dir)
            res.save_to_img(save_path=ocr_result_dir)
            # не работают так как на момент запуска в образе paddleocr-vl ошибка No module named 'docx'
            # res.save_to_html(save_path=ocr_result_dir)
            # res.save_to_xlsx(save_path=ocr_result_dir)
            # res.save_to_word(save_path=ocr_result_dir)

    @staticmethod 
    def paddleocrvl_request(input_file: str | Path, llm_base_url: str) -> None:
        """"
        Отправка запроса на сервер контейнера образа paddleocr-vl
        Взято из документации:
        https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/PaddleOCR-VL.html#43-client-side-invocation
        """
        ocr_result_dir = Path(OCR_RESULT_DIR) / Path(input_file).stem
        ocr_result_dir.mkdir(exist_ok=True, parents=True)
        with open(input_file, 'rb') as file:
            image_bytes = file.read()
            image_base64 = base64.b64encode(image_bytes).decode('ascii')
        # тип файла: 1 - картинка, 0 - PDF
        file_type = 0 if Path(input_file).suffix == '.pdf' else 1
        payload = {
            'file': image_base64,
            'fileType': file_type,
        }
        # этап 1 - предобработка
        response = requests.post(llm_base_url + '/layout-parsing', json=payload)
        response.raise_for_status()
        result = response.json()['result']
        pages = []
        for i, res in enumerate(result['layoutParsingResults']):
            pages.append({'prunedResult': res['prunedResult'], 'markdownImages': res['markdown'].get('images')})
            for img_name, img in res["outputImages"].items():
                img_path = ocr_result_dir / f'{img_name}_{i}.jpg'
                Path(img_path).parent.mkdir(exist_ok=True)
                with open(img_path, 'wb') as f:
                    f.write(base64.b64decode(img))
        logger.info(f'Изображения с аннотациями сохранены в {ocr_result_dir}')
        # этап 2 - OCR
        payload = {
            'pages': pages,
            'concatenatePages': True,
        }
        response = requests.post(llm_base_url + '/restructure-pages', json=payload)
        response.raise_for_status()
        result = response.json()['result']
        res = result['layoutParsingResults'][0]
        md_dir = ocr_result_dir / 'markdown'
        md_dir.mkdir(exist_ok=True)
        md_file = md_dir / 'doc.md'
        md_file.write_text(res['markdown']['text'])
        for img_path, img in res['markdown']['images'].items():
            img_path = md_dir / img_path
            img_path.parent.mkdir(parents=True, exist_ok=True)
            img_path.write_bytes(base64.b64decode(img))
        logger.info(f'Markdown документы сохранены в {md_file}')

    @staticmethod
    def clear_path(path: str | Path) -> None:
        path = Path(path)
        if path.is_dir():
            for file in path.iterdir():
                file.unlink(missing_ok=True)
        else:
            path.unlink(missing_ok=True)

    @staticmethod
    def print_paddleocr_lib_info() -> None:
        """Печатает инфо о библиотеке PaddleOCR"""
        import paddle
        import paddleocr
        
        print(f'PaddleOCR version: {paddleocr.__version__}')
        print(f'Paddle version: {paddle.__version__}')
        print(f'GPU available: {paddle.is_compiled_with_cuda()}')
        print(f'GPU count: {paddle.device.cuda.device_count()}')
