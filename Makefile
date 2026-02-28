PYTHON ?= python3.13

.PHONY: prepare train detect eval record record-negative record-test-positive record-test-negative clean

prepare:
	$(PYTHON) -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt

train:
	.venv/bin/python train.py

detect:
	.venv/bin/python detect.py --debug

eval:
	.venv/bin/python eval.py

record:
	.venv/bin/python record.py --output-dir train_data

record-negative:
	.venv/bin/python record.py --output-dir train_data --prefix neg

record-test-positive:
	.venv/bin/python record.py --output-dir test_data --prefix pos

record-test-negative:
	.venv/bin/python record.py --output-dir test_data --prefix neg

clean:
	rm -rf data/ output/
