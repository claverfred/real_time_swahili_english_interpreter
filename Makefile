.PHONY: install dev lint format test test-all diagnose

install:
	python -m pip install -e ".[local,audio,asr,mt,tts,ui]"

dev:
	python -m pip install -e ".[dev]"

lint:
	ruff check src tests

format:
	ruff format src tests

test:
	pytest -m "not slow and not gpu"

test-all:
	pytest

diagnose:
	python -m src.cli diagnose
