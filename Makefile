PY ?= python3
PACK ?= data/packs/mushroom_body
PORT ?= 8799

.PHONY: help run fetch pack pack-core pack-all test vendor clean

help:
	@echo "make run         serve the viewer on the shipped mushroom-body pack"
	@echo "make fetch       download the FlyWire v783 sources (~880 MB)"
	@echo "make pack        rebuild the shipped mushroom-body pack"
	@echo "make pack-core   build the central-brain pack (61,707 neurons)"
	@echo "make pack-all    build the whole-brain pack (139,248 neurons)"
	@echo "make test        run the test suite"
	@echo "make vendor      re-download three.js into web/vendor"

run:
	$(PY) -m fluctfly.server --pack $(PACK) --port $(PORT) --seed --store data/store

fetch:
	$(PY) -m fluctfly.fetch

pack:
	$(PY) -m fluctfly.build --subset mushroom_body --out data/packs/mushroom_body

pack-core:
	$(PY) -m fluctfly.build --subset core --out data/packs/core

pack-all:
	$(PY) -m fluctfly.build --subset all --out data/packs/all

test:
	$(PY) -m pytest -q

vendor:
	$(PY) scripts/vendor_three.py

clean:
	rm -rf data/raw data/packs/core data/packs/all data/store
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
