.PHONY: install test lint typecheck clean all

all: install test lint typecheck

install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v --tb=short -m "not slow"

test-all:
	pytest tests/ -v --tb=short

lint:
	ruff check src/ tests/

typecheck:
	mypy src/dnnmcshap/

clean:
	rm -rf build/ dist/ *.egg-info src/*.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
