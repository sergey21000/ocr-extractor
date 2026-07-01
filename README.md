

# OCR Extractor

**Извлечение текста из изображений и сканов PDF-документов (без текстового слоя) с использованием Vision Language Models (VLM)**


## 📋 Содержание

- 🏗 [Стек технологий](#-стек-технологий)
- 📁 [Функционал и структура](#-функционал-и-структура)
- 🐳 [Требования](#-требования)
- 🛠 [Подготовка к запуску](#-подготовка-к-запуску)
- 🚀 [Запуск с автоматическим сканированием директории](#-запуск-с-автоматическим-сканированием-директории)
- 🚀 [Запуск с ручной отправкой запросов](#-запуск-с-ручной-отправкой-запросов)
- 🚀 [Запуск клиента и сервера PaddleOCR VL + PaddleOCR vLLM (пример из документации)](#-запуск-клиента-и-сервера-paddleocr-vl--paddleocr-vllm-пример-из-документации)
  - [Запуск](#запуск)
  - [Отправка запросов](#отправка-запросов)
  - [Описание параметров](#описание-параметров)
  - [Описание работы](#описание-работы)
- ⚠ [Решение проблем](#-решение-проблем)


## 🏗 Стек технологий

**Стек технологий**
- [**Python](https://www.python.org/) >= 3.10 - только для отправки OCR запросов
- [**PaddlePaddle/PaddleOCR-VL-1.6**](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6) и [**PaddlePaddle/PaddleOCR-VL-1.6-GGUF**](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6-GGUF) - основная модель для OCR
- [**PaddleOCR**](https://github.com/PaddlePaddle/PaddleOCR) - библиотека и модели для OCR, инференс VLM моделей
- [**llama.cpp**](https://github.com/ggml-org/llama.cpp) - инференс VLM моделей
- [**vLLM**](https://github.com/vllm-project/vllm) - инференс VLM моделей
- [**SGLang**](https://github.com/sgl-project/sglang) - инференс VLM моделей
- [**OpenAI Python**](https://github.com/openai/openai-python) - отправка OCR запросов
- [**Requests**](https://github.com/psf/requests) - отправка OCR запросов
- [**pypdfium2**](https://github.com/pypdfium2-team/pypdfium2) - преобразование PDF в изображения при отправке OCR запросов
- [Pillow](https://github.com/python-pillow/Pillow) - изменение размера изображений перед OCR
- [tqdm](https://github.com/tqdm/tqdm) - прогресс-бар
- [**loguru**](https://github.com/Delgan/loguru) - логгирование


## 📁 Функционал и структура

**Возможны два варианта запуска:**
- запуск сервера с VLM + запуск сервиса который автоматически сканирует директорию `ocr_input` на наличие изображений или PDF, производит процесс распознавания текста и сохраняет результат в директорию `ocr_result` (подходит для локального запуска)
- запуск сервера с VLM + ручная отправка запроса через готовые Python скрипты (подходит для запуска и локально и на удаленном сервере)

**Описание сервисов бэкендов для OCR:**
- `docker/compose.llamacpp.yml` - сервер llama.cpp
- `docker/compose.vllm.yml` - сервер vLLM
- `docker/compose.sglang.yml` - сервер SGLang
- `docker/compose.paddleocr-vlm-server.yml` - сервер PaddleOCR на основе vLLM
- `docker/compose.paddleocr-client-server.yml` - клиент-сервер из [документации](https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/PaddleOCR-VL.html#41-method-1-deploy-using-docker-compose-recommended) PaddleOCR

**Описание сервисов сканеров:**
- `docker/compose.watcher.paddleocrvl.yml` - контейнер с библиотекой и моделями PaddleOCRVL + сканер директории `ocr_input` на наличие изображений или PDF файлов инференс происходит внутри контейнера)
- `docker/compose.watcher.openai.yml` - сервис который сканирует директорию `ocr_input` на наличие изображений или PDF файлов и отправляет OCR запросы на внешний сервис (llama.cpp/vLLM/SGLang/PaddleOCR) через OpeanAI API

**Описание скриптов для OCR запросов:**
- `scripts/request_openai.py` - отправка OCR запросов на серверы llama.cpp / vLLM / SGLang / PaddleOCR
- `scripts/request_paddleocrvl.py` - запуск OCR внутри контейнера PaddleOCRVL
- `scripts/request_paddleocrvl_api.py` - отправка запроса на клиент PaddleOCR из документации

**Описание скриптов сканеров директории:**
- `scripts/run_openai_wather.py` - сканер директории `ocr_input`, запускается в контейнере файла `docker/compose.watcher.openai.yml`
- `scripts/run_paddleocrvl_wather.py` - сканер директории `ocr_input`, запускается в контейнере файла `docker/compose.watcher.paddleocrvl.yml`

**Файлы настроек:**
- `.env` - основной файл настроек для установки переменных окружения для всех сервисов  
В нем прописываются настройки llama.cpp, настройки инициализации моделей PaddleOCRVL, теги для образов PaddleOCR и прочие служебные флаги
- `configs/vllm.yml` - настройка инференса для сервера vLLM / PaddleOCR vLLM
- `configs/sglang.yml` - настройка инференса для сервера SGLang
- `configs/paddleocr_pipeline_vllm.yml` - настройка пайплайна PaddleOCRVL  
(параметры запуска сервера для `docker/compose.paddleocr-vlm-server.yml` и `compose.paddleocr-client-server.yml`)
- `configs/config.py` - настройка пайплайна PaddleOCRVL (параметры инициализации и инференса модели PaddleOCRVL в Python коде), параметры отправки запросов и прочее

В каждом файле настроек есть ссылка на документацию

Модели загружаются в директорию `data/huggingface` - вместо `data` можно указать другую директорию в переменной `DOCKER_VOLUME_STORAGE` в файле `.env`, например 
```env
# пример директории с папкой huggingface, куда загружаются модели HF по умолчанию
DOCKER_VOLUME_STORAGE=C:\Users\<ИМЯ_ПОЛЬЗОВАТЕЛЯ>\.cache
```


## 🐳 Требования

**1) Docker + Docker Compose**

- `Docker Desktop`  
https://docs.docker.com/get-started/introduction/get-docker-desktop/  
или
- `Docker Engine + Docker Compose`  
https://docs.docker.com/engine/install/  
https://docs.docker.com/compose/install/  

Быстрая установка Docker + Docker Compose на Linux
```sh
sudo apt-get update
curl -fsSL https://get.docker.com | sudo sh
sudo systemctl enable --now docker
sudo usermod -aG docker $USER
newgrp docker
```

**2) NVIDIA Container Toolkit (опционально)**

Для работы контейнеров на видеокартах NVIDIA нужно установить NVIDIA Container Toolkit  
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

Быстрая установка NVIDIA Container Toolkit на Linux
```sh
sudo apt-get update && sudo apt-get install -y --no-install-recommends curl gnupg2
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

**3) Python**

Необходим только для ручной отправки OCR запросов  
https://www.python.org/


## 🛠 Подготовка к запуску

**1) Клонирование репозитория**
```ps1
git clone https://github.com/sergey21000/ocr-extractor
cd ocr-extractor
```

**2) Копирование файла `.env` с переменными окружения**
```ps1
cp .env.example .env
```

**3) Установка пути до файла .env**

Установить переменную окружения для env файла, чтобы файлы `docker/compose.yml` видели переменные из файла `.env`

<ins><i>Linux</i></ins>
```
export COMPOSE_ENV_FILES=.env
```

<ins><i>Windows PowerShell</i></ins>
```
$env:COMPOSE_ENV_FILES=".env"
```

**4) Установка библиотек (опционально)**

Установка библиотек для отправки OCR запросов
```
pip install -r requirements.txt
```


## 🚀 Запуск с автоматическим сканированием директории

**Запуск сервера c VLM + сервиса который автоматически сканирует директорию `ocr_input` на наличие изображений или PDF, производит процесс распознавания текста и сохраняет результат в директорию `ocr_result`**

Необходимо выбрать с каким бэкендом запускать из примеров ниже  
После запуска скопировать файлы PDF или изображений в директорию `ocr_input` - начнется процесс OCR  
Результаты сохраняются в директории `ocr_result`  
Данный способ больше подходит для локального запуска  

> [!WARNING]
> Необходимо именно копировать а не перемещать файлы в директорию `ocr_input`, так как после OCR файлы будут удалены из нее (даже если процесс OCR выполнился неудачно)

Параметры инференса `OPENAI_CHAT_COMPLETIONS_KWARGS` и `PADDLEOCRVL_PREDICT_KWARGS` в модуле `configs/config.py` можно редактировать перед каждым запросом даже после запуска сервисов

<ins><i>Запуск c бэкендом llama.cpp</i></ins>
- с поддержкой CUDA
  ```ps1
  docker compose -f docker/compose.llamacpp.yml -f docker/compose.watcher.openai.yml up
  ```
- с поддержкой CPU
  ```ps1
  docker compose -f docker/compose.llamacpp.cpu.yml -f docker/compose.watcher.openai.yml up
  ```
llama.cpp WebUI доступен по адресу http://127.0.0.1:8080  
Можно отправить запрос с текстом `OCR:` и подгрузить картинку прямо в интерфейс

<ins><i>Запуск c бэкендом vLLM</i></ins>
- с поддержкой CUDA
  ```ps1
  docker compose -f docker/compose.vllm.yml -f docker/compose.watcher.openai.yml up
  ```
- с поддержкой CPU
  ```ps1
  docker compose -f docker/compose.vllm.cpu.yml -f docker/compose.watcher.openai.yml up
  ```
vLLM Swagger доступен по адресу http://127.0.0.1:8080/docs

<ins><i>Запуск c бэкендом SGLang</i></ins>
- с поддержкой CUDA
  ```ps1
  docker compose -f docker/compose.sglang.yml -f docker/compose.watcher.openai.yml up
  ```
SGLang Swagger доступен по адресу http://127.0.0.1:8080/docs

<ins><i>Запуск c бэкендом PaddleOCR (vLLM)</i></ins>
- с поддержкой CUDA
  ```ps1
  docker compose -f docker/compose.watcher.paddleocrvl.yml up
  ```
vLLM Swagger доступен по адресу http://127.0.0.1:8080/docs


## 🚀 Запуск с ручной отправкой запросов

**Запуск сервера c VLM с последующей отправкой запроса через Python скрипты или Swagger**

Данный способ подходит для запуска как локально так и на удаленном сервере  
Для удаленного варианта необходимо установить IP адрес сервера в переменную окружения `LLM_BASE_URL=http://<IP>:8080` в файле `.env`  
(WebUI и Swagger будут доступны на http://<IP>:8080 вместо http://127.0.0.1:8080)

<ins><i>Запуск сервера llama.cpp</i></ins>
- с поддержкой CUDA
  ```ps1
  docker compose -f docker/compose.llamacpp.yml up
  ```
- с поддержкой CPU
  ```ps1
  docker compose -f docker/compose.llamacpp.cpu.yml up
  ```
llama.cpp WebUI доступен по адресу http://127.0.0.1:8080  
Можно отправить запрос с текстом `OCR:` и подгрузить картинку прямо в интерфейс

<ins><i>Запуск сервера vLLM</i></ins>
- с поддержкой CUDA
  ```ps1
  docker compose -f docker/compose.vllm.yml up
  ```
- с поддержкой CPU
  ```ps1
  docker compose -f docker/compose.vllm.cpu.yml up
  ```
vLLM Swagger доступен по адресу http://127.0.0.1:8080/docs

<ins><i>Запуск сервера SGLang</i></ins>
- с поддержкой CUDA
  ```ps1
  docker compose -f docker/compose.sglang.yml up
  ```
SGLang Swagger доступен по адресу http://127.0.0.1:8080/docs

<ins><i>Запуск сервера PaddleOCR genai-server (на основе vLLM)</i></ins>
- с поддержкой CUDA
  ```ps1
  docker compose -f docker/compose.paddleocr-vlm-server.yml up
  ```
vLLM Swagger доступен по адресу http://127.0.0.1:8080/docs

---
Отправка OCR запроса
```ps1
python -m scripts.request_openai -i example_files/image_text2.jpg
python -m scripts.request_openai -i "example_files/CT CЭB527-77_527.pdf"
python -m scripts.request_openai -i example_files
```
Результаты сохраняются в директории `ocr_result`


## 🚀 Запуск клиента и сервера PaddleOCR VL + PaddleOCR vLLM (пример из документации)

Данный пример взят из [документации](https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/PaddleOCR-VL.html#41-method-1-deploy-using-docker-compose-recommended) PaddleOCR - это рекомендуемый способ деплоя  
[Документация](https://www.paddleocr.ai/latest/en/version3.x/paddleocr_and_paddlex.html#3-using-paddlex-pipeline-configuration-files) как использовать файлы PaddleOCR  


### Запуск 

<ins><i>Запуск клиента и сервера PaddleOCR (на основе vLLM)</i></ins>
- с поддержкой CUDA
  ```ps1
  docker compose -f docker/compose.paddleocr-client-server.yml up
  ```
PaddleOCR VL Swagger доступен по адресу http://127.0.0.1:8080/docs


### Отправка запросов

<ins><i>Запуск OCR через Python скрипт внутри контейнера PaddleOCR VL</i></ins>  
```
docker exec -it paddleocr-vl-api python -m scripts.request_paddleocrvl -i example_files/image_text2.jpg
docker exec -it paddleocr-vl-api python -m scripts.request_paddleocrvl -i "example_files/CT CЭB527-77_527.pdf"
docker exec -it paddleocr-vl-api python -m scripts.request_paddleocrvl -i example_files

python -m scripts.request_paddleocrvl_api -i example_files/image_text2.jpg
python -m scripts.request_paddleocrvl_api -i "example_files/CT CЭB527-77_527.pdf"
python -m scripts.request_paddleocrvl_api -i example_files
```
Результаты сохраняются в директории `ocr_result`

<ins><i>Запуск OCR через команду терминала `paddleocr-vl doc_parser` внутри контейнера PaddleOCR VL</i></ins>  

Linux терминал
```sh
docker exec -it paddleocr-vl-api paddleocr doc_parser \
	-i /home/paddleocr/example_files/image_text2.jpg \
	--pipeline_version v1.6 \
	--vl_rec_backend vllm-server \
	--vl_rec_server_url http://paddleocr-vlm-server:8080/v1 \
	--vl_rec_api_model_name PaddleOCR-VL-1.6-0.9B \
	--save_path ocr_result/image_text2
```

Windows PowerShell
```ps1
docker exec -it paddleocr-vl-api paddleocr doc_parser `
-i example_files/image_text2.jpg `
--pipeline_version v1.6 `
--vl_rec_backend vllm-server `
--vl_rec_server_url http://paddleocr-vlm-server:8080/v1 `
--vl_rec_api_model_name PaddleOCR-VL-1.6-0.9B `
--save_path ocr_result/image_text2 `
--max_new_tokens 1024
```
Результаты сохраняются в директории `ocr_result`  
Исходные файлы должны находиться в директории `example_files`, поскольку команда запускается из контейнера


### Описание параметров

При передачи изображения с названием `image_text2.jpg` c параметрами по умолчанию будут созданы следующие результаты в директории `ocr_result`
```
image_text2_res.json
image_text2.md
image_text2_layout_det_res.jpg
```

Описание параметров команды `paddleocr doc_parser`  
https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/PaddleOCR-VL.html#21-command-line-usage  
Некоторые параметры:  
- `--use_layout_detection True` - дополнительное обнаружение и классификация фрагментов исходной картинки - будет создано изображение `image_text2_layout_det_res.jpg` с аннотациями (по умолчанию True)
- `--use_doc_unwarping` - отвечает за коррекицю фрагментов изображений с текстом (по умолчанию False)
- `use_doc_orientation_classify` - делать ли классификацию ориентации изображения

Если какая либо команда не была передана - она определяется:
- в конфиге `configs/paddleocr_pipeline_config_vllm.yaml` 
- при инициализации модели PaddleOCRVL (аргументы находятся в `configs/config.py`)
- при инференсе модели PaddleOCRVL (аргументы находятся в `configs/config.py`)
Каждый последующий пункт имеет приоритет над предыдущим

Пример команды без дополнительного обнаружения фрагментов
```sh
docker exec -it paddleocr-vl-api paddleocr doc_parser \
	-i /home/paddleocr/example_files/image_text2.jpg \
	--pipeline_version v1.6 \
	--vl_rec_backend vllm-server \
	--vl_rec_server_url http://paddleocr-vlm-server:8080/v1 \
	--vl_rec_api_model_name PaddleOCR-VL-1.6-0.9B \
	--use_doc_orientation_classify False \
	--use_doc_unwarping False \
	--use_layout_detection False \
	--save_path ocr_result/image_text2 \
	--max_new_tokens 1024
```
С такими параметрами будут созданы следующие результаты в директории `ocr_result`
```
image_text2_res.json
image_text2.md
```
Параметр `--max_new_tokens 1024` установлен таким, чтобы он был не более чем параметр `max_model_len` в `configs/vllm.yml`   
(по умолчанию `--max_new_tokens 4096` но у меня установлен `max_model_len: 2048` для экономии памяти, поэтому пришлось снизить `--max_new_tokens` до 1024)


### Описание работы

<ins><i>Описание работы `docker/compose.paddleocr-client-server.yml`</i></ins>

Сервис `paddleocr-vlm-server` работает как сервер с моделью OCR  
Сервис `paddleocr-vl-api` работает как пайплайн предобработки и для API запросов  

Модель сервера лежит в контейнере paddleocr-genai-vllm-server по пути  
```
/home/paddleocr/.paddlex/official_models/PaddleOCR-VL-1.6/
```
Модели пайплайна лежат в контейнере paddleocr-vl по пути  
```
/home/paddleocr/.paddlex/official_models/
# PP-DocLayoutV3  PP-LCNet_x1_0_doc_ori  PaddleOCR-VL-1.6  UVDoc
```
Описание моделей:  
https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/OCR.html  
- `PP-DocLayoutV3` - Модель для структурного анализа - классификация элементов изображения документов (текст, диаграмма, изображение, таблицы, списки и тд)  
https://www.paddleocr.ai/latest/en/version3.x/module_usage/layout_detection.html
- `PP-LCNet_x1_0_doc_ori` - модель классификации ориентации изображений документов
- `PaddleOCR-VL-1.6` - основная модель OCR
- `UVDoc` - модель коррекции искажений текста и изображений


## ⚠ Решение проблем]

---
**Проблема:**  
При запуске compose файлов на основе образов PaddleOCR  
(`paddleocr-vlm-server.yml`, `paddleocr-client-server.yml`, `compose.paddleocrvl.yml`)  
появляется ошибка `RuntimeError: Unsupported GPU architecture`

**Решение:**  
Необходимо изменить теги образов в файле `.env` в переменных `VLM_IMAGE_TAG_SUFFIX` и/или `API_IMAGE_TAG_SUFFIX`:
- `latest-nvidia-gpu-sm120-offline` для видеокарт архитектуры Blackwell
- `latest-nvidia-gpu-offline` для видеокарт архитектур, отличных от Blackwell

---
**Проблема:**
Нехватка памяти (RAM или VRAM)

**Решения:**
Уменьшить значения параметров в конфигах (ссылки на документации по параметрам находятся в самих конфигах)
- `.env` - llama.cpp, параметры `LLAMA_ARG_CTX_SIZE`, `LLAMA_ARG_N_GPU_LAYERS`, `LLAMA_ARG_CACHE_TYPE_K/V`
- `configs/vllm.yml` - vLLM / PaddleOCR vLLM, параметры `dtype`, `max_model_len`, `max_num_batched_tokens`, `gpu_memory_utilization`, `kv_cache_dtype`, `cpu_offload_gb`, `quantization`
- `configs/sglang.yml` - SGLang, параметры `dtype`, `context-length`, `max-total-tokens`, `mem-fraction-static`,  `kv-cache-dtype`, `cpu_offload_gb`, `cpu-offload-gb`
- `configs/config.py` - PaddleOCRVL, параметр `precision`

---
**Проблема:**
Модель генерирует много повторяющихся слов пордряд (зацикливается)

**Решения:**
Редактировать словарь `OPENAI_CHAT_COMPLETIONS_KWARGS` в модуле `configs/config.py`
- установить параметр `temperature` больше 0 (при этом на [странице](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6-GGUF#basic-usage) модели рекомендуется ставить `temperature=0`)
- увеличить значения параметров, связанные с `_penalty`

---
**Проблема:**
Сервер llama.cpp выдает ошибку
```sh
operator(): got exception: {"error":{"code":500,"message":"The model produced output that does not match the expected peg-native format","type":"server_error"}}
...
common_chat_peg_parse: unparsed peg-native output:
```

**Решение:**
Редактировать словарь `OPENAI_CHAT_COMPLETIONS_KWARGS` в модуле `configs/config.py`
- установить параметр `response_format=dict(type='text')`

---
**Проблема:**
При отправке OCR запросов скрипты выдают ошибку `Error code: 503`, даже если сервер работает на локальной машине

**Решение:**
Отключить VPN и попробовать снова

---
**Проблема:**
При отправке OCR запроса на `docker/compose.paddleocr-client-server.yml` выдает ошибку
```sh
openai.BadRequestError: Error code: 400 - {'error': {'message': "'max_tokens' or 'max_completion_tokens' is too large: 4096. This model's maximum context length is 4096 tokens and your request has 14 input tokens (4096 > 4096 - 14). None", 'type': 'BadRequestError', 'param': None, 'code': 400}}
```

**Решение:**
Параметр `max_model_len` в `configs/vllm.yml` должен быть больше чем параметр `PADDLEOCRVL_PREDICT_KWARGS['max_new_tokens']` в `configs/config.py` - например `max_model_len: 8192`
В данном примере ошибки они установлены так - что приводит к ошибке
```
# configs/vllm.yml
max_model_len: 4096

# configs/config.py
PADDLEOCRVL_PREDICT_KWARGS = dict(
    max_new_tokens=4096,  # default: 4096
)
```
В данном репозитории установлен `max_model_len: 2048` для экономии памяти, поэтому пришлось снизить `max_new_tokens` до 1024 - можно увеличивать эти значения если ресурсов достаточно

