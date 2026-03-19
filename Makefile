.PHONY: install test lint format demo clean

PYTHON ?= $(HOME)/anaconda3/bin/python

install:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest tests/ -v

lint:
	$(PYTHON) -m flake8 src/ tests/ --max-line-length=120 || true
	$(PYTHON) -m mypy src/ --ignore-missing-imports || true

format:
	$(PYTHON) -m black src/ tests/ || true
	$(PYTHON) -m isort src/ tests/ || true

demo:
	$(PYTHON) -m modelcard_generator generate examples/bert_sentiment.yaml
	@echo ""
	@echo "--- Drift check (simulating metric regression) ---"
	$(PYTHON) -m modelcard_generator generate examples/bert_sentiment.yaml -o /tmp/demo_card.md
	$(PYTHON) -m modelcard_generator check /tmp/demo_card.md --new-eval examples/eval_results_v2.json || true

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache dist build *.egg-info src/*.egg-info
