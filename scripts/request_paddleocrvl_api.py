# https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/PaddleOCR-VL.html#43-client-side-invocation

import os
import argparse
from pathlib import Path

from configs.config import EXAMPLE_IMAGE_PATH, EXAMPLE_PDF_PATH
from utils.paddle import OcrUtilsPaddle


def main():
    LLM_BASE_URL = os.getenv('LLM_BASE_URL', 'http://127.0.0.1:8080')
    if LLM_BASE_URL.endswith('/v1'):
        LLM_BASE_URL = LLM_BASE_URL[:-3]
    parser = argparse.ArgumentParser(description='OCR')
    parser.add_argument('-i', '--input', type=str, required=False, help='Путь к директории/файлу (PDF, изображения)')
    args = parser.parse_args()
    input_path = args.input
    if not input_path:
        files = [Path(EXAMPLE_IMAGE_PATH), Path(EXAMPLE_PDF_PATH)]
    else:
        input_path = Path(input_path)
        if input_path.is_dir():
            files = [f for f in input_path.iterdir() if f.suffix.lower() in OcrUtilsPaddle.SUPPORTED_EXTENSIONS]
        else:
            files = [input_path]
    for file_path in files:
        OcrUtilsPaddle.paddleocrvl_request(input_file=file_path, llm_base_url=LLM_BASE_URL)


if __name__ == '__main__':
    main()
