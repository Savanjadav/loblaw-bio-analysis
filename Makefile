PYTHON ?= python3

.PHONY: setup pipeline dashboard test

setup:
	$(PYTHON) -m pip install -r requirements.txt

pipeline:
	$(PYTHON) load_data.py
	MPLCONFIGDIR="$${TMPDIR:-/tmp}/loblaw-matplotlib" $(PYTHON) analysis.py

dashboard:
	MPLCONFIGDIR="$${TMPDIR:-/tmp}/loblaw-matplotlib" $(PYTHON) -m streamlit run dashboard.py

test:
	MPLCONFIGDIR="$${TMPDIR:-/tmp}/loblaw-matplotlib" $(PYTHON) -m pytest -q
