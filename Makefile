PYTHON ?= python3.13

.PHONY: prepare train detect eval record-positive record-negative record-test-positive record-test-negative sort-records clean

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

record-positive:
	.venv/bin/python record.py --output-dir train_data/positive --prefix pos

record-negative:
	.venv/bin/python record.py --output-dir train_data/negative --prefix neg

record-test-positive:
	.venv/bin/python record.py --output-dir test_data --prefix pos

record-test-negative:
	.venv/bin/python record.py --output-dir test_data --prefix neg

sort-records:
	.venv/bin/python sort_recordings.py

clean:
	rm -rf data/ output/
