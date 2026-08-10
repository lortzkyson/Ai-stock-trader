.PHONY: install test lint typecheck check data-quality train backtest monitor performance-report

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

data-quality:
	$(VENV)/python scripts/check_data_quality.py data/cache/1Min/*.parquet

train:
	$(VENV)/python scripts/fetch_training_data.py
	$(VENV)/python scripts/train_model.py

backtest:
	$(VENV)/python scripts/run_backtest.py

monitor:
	$(VENV)/python scripts/monitor.py

performance-report:
	$(VENV)/python scripts/generate_performance_report.py

# Kill switch takes an action argument, so call it directly rather than via make:
#   .venv/bin/python scripts/kill_switch.py {status,engage,disengage} [--flatten]
