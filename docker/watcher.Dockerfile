FROM ghcr.io/astral-sh/uv:python3.12-trixie-slim
WORKDIR /app
ENV UV_SYSTEM_PYTHON=1
COPY requirements.txt .
RUN uv pip install --no-cache-dir -r requirements.txt
COPY utils utils
COPY scripts scripts
COPY configs configs
CMD ["uv", "run", "-m", "scripts.run_openai_wather"]
