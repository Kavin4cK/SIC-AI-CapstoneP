.PHONY: dev test build seed clean

dev:
	@bash ./run.sh

test:
	@PYTHONPATH=. .venv/bin/pytest dashboard/tests/test_api.py

test-all:
	@PYTHONPATH=. .venv/bin/pytest tests/ dashboard/tests/

build:
	@cd dashboard/frontend && npm run build

seed:
	@PYTHONPATH=. .venv/bin/python src/predict.py --input outputs/predictions/sample_input.csv --output outputs/predictions/flagged_transactions.csv --threshold 0.144

clean:
	@rm -rf dashboard/frontend/dist dashboard/data/*.sqlite3*
