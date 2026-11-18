.PHONY: all data analysis test lint
all: data analysis test

data:        ## download play-by-play (skips files you already have)
	python -m scripts.download_data

refresh:     ## re-download, for in-season updates
	python -m scripts.download_data --force

analysis:    ## train, evaluate, write charts and findings.json
	python -m gamescript.analysis

test:
	pytest -q

lint:
	ruff check .
