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

import requests
import pypdfium2 as pdfium
from PIL import Image
from paddleocr import PaddleOCRVL

from utils.logger import logger
from utils.base import OcrBaseUtils
from configs.config import (
    PADDLEOCRVL_INIT_KWARGS,
    OCR_RESULT_DIR,
)


class OcrUtilsPaddle(OcrBaseUtils):
    """Утилиты для отправки запросов через OpenAI"""
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

    @classmethod
    def paddleocrvl_request(cls, input_file: str | Path, llm_base_url: str) -> None:
        """"
        Отправка HTTP API запроса на сервер контейнера образа PaddleOCR-VL 
        Взято из документации:
        https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/PaddleOCR-VL.html#43-client-side-invocation
        """
        import importlib
        import configs.config
        importlib.reload(configs.config)
        from configs.config import PADDLEOCRVL_PREDICT_KWARGS, OCR_MAX_IMAGE_DIMENSION

        def clean(d: dict) -> dict:
            return {k: v for k, v in d.items() if v is not None}

        def to_camel(d: dict) -> dict:
            # snake_case -> camelCase
            mapping = {
                'use_doc_orientation_classify': 'useDocOrientationClassify',
                'use_doc_unwarping': 'useDocUnwarping',
                'use_layout_detection': 'useLayoutDetection',
                'use_chart_recognition': 'useChartRecognition',
                'use_seal_recognition': 'useSealRecognition',
                'repetition_penalty': 'repetitionPenalty',
                'top_p': 'topP',
                'max_new_tokens': 'maxNewTokens',
                'min_pixels': 'minPixels',
                'max_pixels': 'maxPixels',
                'layout_nms': 'layoutNms',
                'layout_unclip_ratio': 'layoutUnclipRatio',
                'layout_merge_bboxes_mode': 'layoutMergeBboxesMode',
                'format_block_content': 'formatBlockContent',
                'merge_layout_blocks': 'mergeLayoutBlocks',
                'markdown_ignore_labels': 'markdownIgnoreLabels',
            }
            return {mapping.get(k, k): v for k, v in d.items()}

        extra_kwargs = PADDLEOCRVL_PREDICT_KWARGS
        ocr_result_dir = Path(OCR_RESULT_DIR) / Path(input_file).stem
        ocr_result_dir.mkdir(exist_ok=True, parents=True)

        # готовим содержимое файла с учётом ресайза
        input_file = Path(input_file)
        if input_file.suffix.lower() == '.pdf':
            with pdfium.PdfDocument(input_file, password=None) as src_pdf:
                pil_images = [
                    page.render(scale=1, rev_byteorder=True).to_pil()
                    for page in src_pdf
                ]
            file_bytes = cls.build_resized_pdf(pil_images, OCR_MAX_IMAGE_DIMENSION)
            file_type = 0
        else:
            pil_image = Image.open(input_file)
            if OCR_MAX_IMAGE_DIMENSION:
                pil_image = cls.resize_pil_image(pil_image, OCR_MAX_IMAGE_DIMENSION)
            buf = BytesIO()
            pil_image.convert('RGB').save(buf, format='JPEG')
            file_bytes = buf.getvalue()
            file_type = 1

        image_base64 = base64.b64encode(file_bytes).decode('ascii')

        # 1) layout-parsing
        layout_kwargs = clean(to_camel({
            k: extra_kwargs.get(k)
            for k in [
                'use_doc_orientation_classify',
                'use_doc_unwarping',
                'use_layout_detection',
                'layout_nms',
                'layout_unclip_ratio',
                'layout_merge_bboxes_mode',
                'format_block_content',
                'temperature',
                'top_p',
                'repetition_penalty',
                'max_new_tokens',
            ]
        }))
        payload = {
            'file': image_base64,
            'fileType': file_type,
            **layout_kwargs,
        }
        response = requests.post(
            f'{llm_base_url}/layout-parsing',
            json=payload
        )
        response.raise_for_status()
        result = response.json()['result']
        pages = []
        for i, res in enumerate(result['layoutParsingResults']):
            pages.append({
                'prunedResult': res['prunedResult'],
                'markdownImages': res['markdown'].get('images'),
            })
            for img_name, img in res.get('outputImages', {}).items():
                img_path = ocr_result_dir / f'{img_name}_{i}.jpg'
                img_path.parent.mkdir(parents=True, exist_ok=True)
                img_path.write_bytes(base64.b64decode(img))
        logger.info(f'Изображения с аннотациями сохранены в {ocr_result_dir}')

        # 2) restructure-pages
        vlm_kwargs = clean(to_camel({
            k: extra_kwargs.get(k)
            for k in [
                'merge_layout_blocks',
                'markdown_ignore_labels',
            ]
        }))
        payload = {
            'pages': pages,
            'concatenatePages': True,
            **vlm_kwargs,
        }
        response = requests.post(
            f'{llm_base_url}/restructure-pages',
            json=payload
        )
        response.raise_for_status()
        result = response.json()['result']
        res = result['layoutParsingResults'][0]
        md_file = ocr_result_dir / Path(input_file.name).with_suffix('.md')
        md_file.write_text(res['markdown']['text'], encoding='utf-8')
        for img_path, img in res['markdown'].get('images', {}).items():
            full_path = ocr_result_dir / img_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_bytes(base64.b64decode(img))
        logger.info(f'Markdown документы сохранены в {ocr_result_dir}')

    @staticmethod
    def print_paddleocr_lib_info() -> None:
        """Печатает инфо о библиотеке PaddleOCR"""
        import paddle
        import paddleocr
        
        print(f'PaddleOCR version: {paddleocr.__version__}')
        print(f'Paddle version: {paddle.__version__}')
        print(f'GPU available: {paddle.is_compiled_with_cuda()}')
        print(f'GPU count: {paddle.device.cuda.device_count()}')
