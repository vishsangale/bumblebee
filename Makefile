PYTHON ?= python3

.PHONY: setup test lint format check

setup:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check src tests

format:
	ruff format src tests

check: lint test
