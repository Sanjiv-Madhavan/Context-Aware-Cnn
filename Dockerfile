FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.lock.txt requirements.txt ./
RUN pip install --upgrade pip && \
    if [ -f requirements.lock.txt ]; then pip install -r requirements.lock.txt; else pip install -r requirements.txt; fi

COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs

RUN pip install -e .

ENTRYPOINT ["python", "-m", "unet_denoising.cli"]
CMD ["--help"]
