.PHONY: install viz test train inf eval lint format clean all

PYTHON = $(shell which python)
UV = $(shell which uv)
PIP = $(shell which pip)

all:
	train inf

install:
	@echo "Creating a virtual environment ..."
	$(UV) venv
	@echo "Installing the project in dev mode..."
	source .env/bin/activate
	$(UV) $(PIP) install -r requirements-dev.txt
	@echo "Environment ready to be used."

test:
	source .env/bin/activate
	PYTHONPATH=src $(UV) run pytest tests/ -v

viz:
	$(UV) run python -m visualize

train:
	$(UV) run python -m train

inf:
	$(UV) run python -m inference

eval:
	$(uv) run python -m evaluate

lint: 
	$(UV) run ruff check src/ tests/

format:
	$(UV) run black src/ tests/
	$(UV) run isort src/ tests/

clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type d -name ".pytest_cache" -exec rm -r {} +
	find . -type d -name "ter_project.egg-info" -exec rm -fr {} +
	rm -fr src/*.pkl src/*.pyc tests/*.pkl tests/*.pyc notebooks/.ipynb_checkpoints .vscode .coverage*



