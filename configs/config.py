import os


# описание параметров инициализации PaddleOCRVL (под спойлером)
# https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/PaddleOCR-VL.html#21-command-line-usage
# исходники класса PaddleOCRVL:
# https://github.com/PaddlePaddle/PaddleOCR/blob/release/3.7/paddleocr/_pipelines/paddleocr_vl.py#L37
PADDLEOCRVL_INIT_KWARGS = dict(
    # отключить допольнительный препроцессинг чтобы было больше свободной памяти
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_layout_detection=False,
    # fp16, fp32 (default)
    precision='fp16',
)

# исходники метода predict класса PaddleOCRVL:
# https://github.com/PaddlePaddle/PaddleOCR/blob/release/3.7/paddleocr/_pipelines/paddleocr_vl.py#L157
PADDLEOCRVL_PREDICT_KWARGS = dict(
    use_doc_orientation_classify=None,
    use_doc_unwarping=None,
    use_layout_detection=None,
    repetition_penalty=None,
    temperature=None,
    top_p=None,
    max_new_tokens=1024,  # default: 4096
    # min_pixels=None,
    # max_pixels=None,
)

# пример:
# PADDLEOCRVL_PREDICT_KWARGS = dict(
    # use_doc_orientation_classify=True,
    # use_doc_unwarping=False,
    # use_layout_detection=True,
    # repetition_penalty=1.1,
    # temperature=0.25,
    # top_p=0.9,
    # min_pixels=480,
    # max_pixels=1080,
    # max_new_tokens=1024,
    # use_tensorrt=False,
    # enable_hpi=True,
# )

# параметры для запроса client.chat.completions.create()
OPENAI_CHAT_COMPLETIONS_KWARGS = dict(
    temperature=0.1,
    max_tokens=512,
    response_format=dict(type='text'),
    extra_body=dict(
        repeat_penalty=1,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        # top_k=20,
        # top_p=0.9,
    )
)
# интервал сканирования директории с картинками и PDF
SCAN_INTERVAL_SECONDS = 5
# поддерживаемые форматы файлов
SUPPORTED_IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
SUPPORTED_PDF_EXTENSIONS = ['.pdf']
# изменение размера изображения перед OCR
# установить None чтобы не применять
OCR_MAX_IMAGE_DIMENSION = 800
# директория для сканирования
OCR_INPUT_DIR = 'ocr_input'
# директория для результатов OCR
OCR_RESULT_DIR = 'ocr_result'
# файлы для проверки OCR по умолчанию (если скрипты запускаются без параметра -i)
EXAMPLE_IMAGE_PATH = 'example_files/image_text2.jpg'
EXAMPLE_PDF_PATH = 'example_files/CT CЭB527-77_527.pdf'
# уровень логгирования - если его нет в ENV то устанавливается тот что написан здесь
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
