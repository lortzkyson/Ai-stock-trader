.PHONY: install test lint typecheck check backtest paper-trade

VENV := .venv/bin

install:
	python3 -m venv .venv
	$(VENV)/pip install --upgrade pip
	$(VENV)/pip install -r requirements.txt

test:
	$(VENV)/pytest --cov=src --cov-report=term-missing

lint:
	$(VENV)/ruff check src tests

typecheck:
	$(VENV)/mypy src

check: lint typecheck test

backtest:
	$(VENV)/python -m backtest.run

paper-trade:
	$(VENV)/python -m execution.run --mode paper
