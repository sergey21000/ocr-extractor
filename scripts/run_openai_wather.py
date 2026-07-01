import os
import time
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

from utils.logger import logger
from utils.openai import OcrUtilsOpenai
from configs.config import (
    OCR_INPUT_DIR,
    OCR_RESULT_DIR,
    SCAN_INTERVAL_SECONDS,
)

LLM_BASE_URL = os.getenv('LLM_BASE_URL', 'http://127.0.0.1:8080/v1')
client = OpenAI(
    api_key='-',
    base_url=LLM_BASE_URL,
    timeout=3600,
)
ocr_input_dir = Path(OCR_INPUT_DIR)
ocr_input_dir.mkdir(exist_ok=True, parents=True)
logger.info(f'Сканер директории {ocr_input_dir} запущен')

while True:
    docs = [f for f in ocr_input_dir.iterdir() if f.suffix.lower() in OcrUtilsOpenai.SUPPORTED_EXTENSIONS]
    if docs:
        pbar = tqdm(total=len(docs), desc='Детекция документов', leave=False)
        for doc_path in docs:
            try:
                result_ocr_texts = OcrUtilsOpenai.ocr_document(client, doc_path)
                if result_ocr_texts:
                    text_for_write = '/n/n'.join(result_ocr_texts)
                    output_file = Path(OCR_RESULT_DIR) / Path(doc_path.stem) / Path(doc_path.name).with_suffix('.txt')
                    output_file.parent.mkdir(exist_ok=True, parents=True)
                    output_file.unlink(missing_ok=True)
                    output_file.write_text(text_for_write, encoding='utf-8')
                    logger.info(f'Результат OCR для документа {doc_path} сохранен в {output_file}')
                else:
                    logger.error('Функция `openai_ocr_document()` вернула пустой список')
            except Exception as e:
                logger.exception(f'Ошибка вызова `openai_ocr_document()`: {e}')
            doc_path.unlink(missing_ok=True)
            pbar.update()
    time.sleep(SCAN_INTERVAL_SECONDS)
