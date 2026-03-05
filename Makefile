.PHONY: lock install-dev test coverage build run-download run-s3-push run-s3-pull run-train run-infer run-sweep-optuna docker-build

PYTHON ?= .venv/bin/python
PIP ?= .venv/bin/pip
PROJECT_ROOT ?= $(PWD)

lock:
	pip-compile requirements.txt -o requirements.lock.txt

install-dev:
	$(PIP) install -e ".[dev]"

test:
	$(PYTHON) -m pytest -q

coverage:
	$(PYTHON) -m pytest --cov=src/unet_denoising --cov-report=term-missing --cov-report=xml --cov-report=html -q
	$(PYTHON) -m webbrowser "file://$(PWD)/htmlcov/index.html"

build:
	$(PYTHON) -m build

run-train:
	PROJECT_ROOT=$(PROJECT_ROOT) $(PYTHON) -m unet_denoising.cli --config configs/config.yaml train

run-infer:
	PROJECT_ROOT=$(PROJECT_ROOT) $(PYTHON) -m unet_denoising.cli --config configs/config.yaml infer

run-download:
	PROJECT_ROOT=$(PROJECT_ROOT) $(PYTHON) -m unet_denoising.cli --config configs/config.yaml download

run-s3-push:
	PROJECT_ROOT=$(PROJECT_ROOT) $(PYTHON) -m unet_denoising.cli --config configs/config.yaml s3-push

run-s3-pull:
	PROJECT_ROOT=$(PROJECT_ROOT) $(PYTHON) -m unet_denoising.cli --config configs/config.yaml s3-pull

run-sweep-optuna:
	PROJECT_ROOT=$(PROJECT_ROOT) $(PYTHON) -m unet_denoising.cli --config configs/config.yaml sweep-optuna

docker-build:
	docker build -t unet-denoising-prod:latest .
