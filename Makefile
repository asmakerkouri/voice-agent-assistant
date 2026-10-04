install:
	pip install -r requirements-dev.txt
	./scripts/download_models.sh

test:
	ruff check . && pytest -q

eval:            ## WER, intent accuracy and latency benchmark + figures (≈ 5 min on CPU)
	PYTHONPATH=src python eval/run_eval.py
	python scripts/make_figures.py

demo:
	PYTHONPATH=src python -m voiceagent.cli --say "There is a crack on wall twelve, level two. It looks major."

serve:
	PYTHONPATH=src uvicorn voiceagent.api:app --port 8000

.PHONY: install test eval demo serve
