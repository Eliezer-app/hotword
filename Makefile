.PHONY: setup train detect eval record record-negative record-test-positive record-test-negative clean

setup:
	pip install -r requirements.txt

train:
	python train.py

detect:
	python detect.py --debug

eval:
	python eval.py

record:
	python record.py --output-dir train_data

record-negative:
	python record.py --output-dir train_data --prefix neg

record-test-positive:
	python record.py --output-dir test_data --prefix pos

record-test-negative:
	python record.py --output-dir test_data --prefix neg

clean:
	rm -rf data/ output/
