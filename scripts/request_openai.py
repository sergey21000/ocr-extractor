import os
import argparse
from pathlib import Path

from openai import OpenAI

from utils.logger import logger
from utils.openai import OcrUtilsOpenai
from configs.config import OCR_RESULT_DIR, EXAMPLE_IMAGE_PATH, EXAMPLE_PDF_PATH


def main():
    LLM_BASE_URL = os.getenv('LLM_BASE_URL', 'http://127.0.0.1:8080/v1')
    client = OpenAI(
        api_key='-',
        base_url=LLM_BASE_URL,
        timeout=3600,
    )
    parser = argparse.ArgumentParser(description='OCR')
    parser.add_argument('-i', '--input', type=str, required=False, help='Путь к директории/файлу (PDF, изображения)')
    args = parser.parse_args()
    input_path = args.input
    if not input_path:
        files = [Path(EXAMPLE_IMAGE_PATH), Path(EXAMPLE_PDF_PATH)]
    else:
        input_path = Path(input_path)
        if input_path.is_dir():
            files = [f for f in input_path.iterdir() if f.suffix.lower() in OcrUtilsOpenai.SUPPORTED_EXTENSIONS]
        else:
            files = [input_path]
    for file_path in files:
        result_ocr_texts = OcrUtilsOpenai.ocr_document(client, file_path)
        if result_ocr_texts:
            text_for_write = '/n/n'.join(result_ocr_texts)
            output_file = Path(OCR_RESULT_DIR) / Path(file_path.stem) / Path(file_path.name).with_suffix('.txt')
            output_file.parent.mkdir(exist_ok=True, parents=True)
            output_file.unlink(missing_ok=True)
            output_file.write_text(text_for_write, encoding='utf-8')
            logger.info(f'Результат OCR для документа {file_path} сохранен в {output_file}')


if __name__ == '__main__':
    main()
