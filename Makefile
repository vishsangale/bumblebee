PYTHON ?= python3

.PHONY: setup test lint format check show-config smoke

setup:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check src tests

format:
	ruff format src tests

check: lint test

show-config:
	$(PYTHON) experiments/train.py --cfg job --resolve

smoke:
	$(PYTHON) experiments/train.py trainer.max_epochs=1 experiment.num_train=64 experiment.num_val=32
