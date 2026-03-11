PYTHON ?= python3.13

.PHONY: prepare train train-e2e detect detect-e2e eval eval-e2e record-positive record-negative record-test-positive record-test-negative sort-records clean

prepare:
	$(PYTHON) -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt

train:
	.venv/bin/python train.py

detect:
	.venv/bin/python detect.py --debug

extract-e2e:
	.venv/bin/python train_e2e.py --extract --aug 10 --max-neg 500

train-e2e:
	.venv/bin/python train_e2e.py --seeds 5 --epochs 80 --aug 10 --max-neg 500

detect-e2e:
	.venv/bin/python detect_e2e.py --debug

eval:
	.venv/bin/python eval.py

eval-e2e:
	.venv/bin/python eval_e2e.py

record-positive:
	.venv/bin/python record.py --output-dir train_data/positive --prefix pos

record-negative:
	.venv/bin/python record.py --output-dir train_data/negative --prefix neg

record-test-positive:
	.venv/bin/python record.py --output-dir test_data/positive --prefix pos

record-test-negative:
	.venv/bin/python record.py --output-dir test_data/negative --prefix neg

sort-records:
	.venv/bin/python sort_recordings.py

clean:
	rm -rf data/ output/
