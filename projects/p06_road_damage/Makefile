# Project 6: road damage detection on RDD2022. Run from this folder.
PYTHON ?= python
SOURCE ?= data/raw
SMOKE_DIR := artifacts/smoke

.PHONY: install data train evaluate export report all demo smoke test lint clean

install:             ## this project with the demo extra and the shared common package
	$(PYTHON) -m pip install -e ../../common -e ".[dev,demo]"

data:                ## RDD2022_Czech.zip and RDD2022_India.zip in $(SOURCE) -> audit tables and a YOLO dataset
	$(PYTHON) -m road_damage.data --source $(SOURCE)

train:               ## three runs of YOLO11n on CPU: frozen backbone, full fine tune, light augmentation
	$(PYTHON) -m road_damage.train

evaluate:            ## test split mAP per run, per class, per country and per lighting bin; ONNX export and CPU speed
	$(PYTHON) -m road_damage.evaluate
	$(PYTHON) -m road_damage.export

report:              ## data audit, tables, figures and results.json under reports/
	$(PYTHON) -m road_damage.report

all: data train evaluate report

demo:                ## Gradio upload demo on http://127.0.0.1:7860
	$(PYTHON) -m road_damage.app

smoke:               ## the whole pipeline on synthetic zips, from untrained weights; outputs stay under artifacts/smoke
	$(PYTHON) -m road_damage.synthetic $(SMOKE_DIR)/raw
	$(PYTHON) -m road_damage.data --source $(SMOKE_DIR)/raw --data-dir $(SMOKE_DIR)/data --block 4
	$(PYTHON) -m road_damage.train --data $(SMOKE_DIR)/data/yolo/dataset.yaml --artifacts $(SMOKE_DIR) \
		--weights yolo11n.yaml --epochs 1 --imgsz 160 --batch 8 --workers 0
	$(PYTHON) -m road_damage.evaluate --yolo-dir $(SMOKE_DIR)/data/yolo --artifacts $(SMOKE_DIR) --imgsz 160 --batch 8
	$(PYTHON) -m road_damage.export --yolo-dir $(SMOKE_DIR)/data/yolo --artifacts $(SMOKE_DIR) --imgsz 160 --images 10
	$(PYTHON) -m road_damage.report --data-dir $(SMOKE_DIR)/data --artifacts $(SMOKE_DIR) --reports $(SMOKE_DIR)/reports

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check src tests
	$(PYTHON) -m ruff format --check src tests

clean:
	rm -rf artifacts
