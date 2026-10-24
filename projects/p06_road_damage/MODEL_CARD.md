# Model card: road damage detection (project 6)

## Model

Ultralytics YOLO11n (2.6 million parameters), started from the COCO weights of the Ultralytics GitHub release and fine tuned on all layers for 12 epochs at 512 pixels, batch 16, seed 42, with light augmentation (no mosaic; scale 0.2, saturation 0.3, value 0.2, translation 0.05; default horizontal flip). Output: boxes with one of four damage types and a confidence. Files: `artifacts/runs/light_augment/weights/best.pt` (PyTorch) and `artifacts/road_damage_yolo11n.onnx` (ONNX, 10.1 MB, input 512 x 512).

| Damage type | Code |
| --- | --- |
| longitudinal crack | D00 |
| transverse crack | D10 |
| alligator crack | D20 |
| pothole | D40 |

Training: Google Colab, Tesla T4 GPU, PyTorch 2.11.0, Ultralytics 8.4.184, 17.1 minutes. Evaluation and export: CPU, PyTorch 2.14.1, ONNX Runtime 1.31.0.

## Intended use

A portfolio demonstration of object detection: transfer learning, an augmentation study, error analysis by subgroup and CPU deployment. **Not intended for road maintenance decisions.** It has not been tested on images from other countries, cameras or seasons.

## Training data

RDD2022 (Arya et al.; images under CC BY-SA 4.0 per the RoadDamageDetector README), labelled images of Czech Republic and India: 10,535 images, split 70/15/15 per country by blocks of 25 consecutive file names (7,360 train, 1,600 validation, 1,575 test). Labels other than the four damage types (for example D44 and D43 in India) are dropped. About 60 percent of the images contain no scored damage.

## Metrics on the held out test split

mAP from all detections down to confidence 0.001; precision and recall at the confidence that maximizes F1.

| Metric | Value |
| --- | ---: |
| mAP50 | 0.288 |
| mAP50-95 | 0.121 |
| Precision | 0.341 |
| Recall | 0.332 |
| AP50 D00 / D10 / D20 / D40 | 0.272 / 0.089 / 0.512 / 0.279 |
| mAP50, ONNX model | 0.289 |
| CPU time per image, ONNX Runtime (median, 4 cores) | 26.6 ms |

Baseline (frozen COCO backbone): mAP50 0.214. Full fine tune with default augmentation: 0.261.

## Subgroups

| Subset | Images | Boxes | mAP50 |
| --- | ---: | ---: | ---: |
| Czech | 425 | 222 | 0.189 |
| India | 1,150 | 1,083 | 0.270 |
| bright | 508 | 684 | 0.307 |
| medium | 528 | 344 | 0.275 |
| dark | 539 | 277 | 0.290 |

## Errors

Confusion matrix on the test split at confidence 0.25 (the default of the demo), counted by Ultralytics with IoU 0.45 ([figure](reports/figures/confusion_matrix.png)):

| True type | Boxes | Found with the right type | Found as another type | Missed |
| --- | ---: | ---: | ---: | ---: |
| D00 | 367 | 57 | 3 | 307 |
| D10 | 69 | 0 | 1 | 68 |
| D20 | 341 | 152 | 1 | 188 |
| D40 | 528 | 73 | 3 | 452 |

Detections without a true box: 157 (41 D00, 78 D20, 38 D40). The model rarely confuses one damage type with another; its main error is missed damage. At 0.25 it misses most cracks and potholes and never predicts D10, so a user who needs recall should lower the confidence slider of the demo.

## Limits

* Two countries only. Japan, Norway, the United States and China are in RDD2022 but not in this model.
* 12 epochs; validation mAP still rose at the last epoch, so the model is undertrained.
* Transverse cracks (D10) have few examples (467 boxes, 68 in India) and AP50 below 0.1.
* The split keeps blocks of consecutive images together, but images of the same road can still appear in more than one block, so the test score may be optimistic.
* The RDD2022 test images have no public labels, so the scores are not comparable with the CRDDC2022 leaderboard.
