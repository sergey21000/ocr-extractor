import time
from pathlib import Path

from utils.logger import logger
from utils.paddle import OcrUtilsPaddle
from configs.config import (
    SCAN_INTERVAL_SECONDS,
    OCR_INPUT_DIR,
)


logger.info('Инициализация модели PaddleOCRVL')

ocr_input_dir = Path(OCR_INPUT_DIR)
ocr_input_dir.mkdir(exist_ok=True, parents=True)
pipeline = OcrUtilsPaddle.get_paddleocrvl_pipepline()

logger.info(f'Сканер директории {ocr_input_dir} запущен')

while True:
    dir_not_empty = bool(list(ocr_input_dir.iterdir()))
    if dir_not_empty:
        bad_files = [
            f for f in ocr_input_dir.iterdir() 
            if not f.is_file() 
            or f.suffix.lower() not in OcrUtilsPaddle.SUPPORTED_EXTENSIONS
        ]
        if bad_files:
            logger.error(
                f'Неподдерживаемые файлы: {bad_files}' 
                '\nФормат должен быть один из: {OcrUtilsPaddle.SUPPORTED_EXTENSIONS}'
            )
            for file in bad_files:
                file.unlink(missing_ok=True)
            time.sleep(SCAN_INTERVAL_SECONDS)
            continue
        try:
            OcrUtilsPaddle.paddleocrvl_predict_and_save(pipeline=pipeline, input_path=ocr_input_dir)
        except Exception as e:
            logger.exception(f'Ошибка вызова `paddleocrvl_predict_and_save()`: {e}')
        OcrUtilsPaddle.clear_path(path=ocr_input_dir)
    time.sleep(SCAN_INTERVAL_SECONDS)
