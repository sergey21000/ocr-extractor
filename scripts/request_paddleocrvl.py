import argparse
import shutil
from pathlib import Path
import tempfile

from configs.config import EXAMPLE_IMAGE_PATH, EXAMPLE_PDF_PATH
from utils.paddle import OcrUtilsPaddle


def main():
    parser = argparse.ArgumentParser(description='OCR')
    parser.add_argument(
        '-i',
        '--input',
        type=str,
        required=False,
        help='Путь к директории/файлу (PDF, изображения)',
    )
    parser.add_argument(
        '-b',
        '--vl_rec_backend',
        type=str,
        required=False,
        choices=[
            'vllm-server',
            'llama-cpp-server',
            'sglang-server',
            'fastdeploy-server',
            'mlx-vlm-server',
            'local',
        ],
        default=None,
        help='PaddleOCRVL recognition backend',
    )
    args = parser.parse_args()
    input_path = args.input
    if not input_path:
        input_path = Path(tempfile.mkdtemp())
        shutil.copy(Path('example_files') / EXAMPLE_IMAGE_PATH,  input_path / EXAMPLE_IMAGE_PATH)
        shutil.copy(Path('example_files') / EXAMPLE_PDF_PATH,  input_path / EXAMPLE_PDF_PATH)
    pipeline = OcrUtilsPaddle.get_paddleocrvl_pipepline(vl_rec_backend=args.vl_rec_backend)
    OcrUtilsPaddle.paddleocrvl_predict_and_save(pipeline=pipeline, input_path=input_path)


if __name__ == '__main__':
    main()
