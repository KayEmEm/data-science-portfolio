# Project 6: Road damage detection

Detect and classify road surface damage (longitudinal, transverse and alligator cracks, and potholes) in street images, so that a city can rank repairs.

**Status: complete.** Main model: YOLO11n fine tuned with light augmentation, **test mAP50 0.288** (frozen backbone baseline 0.214). ONNX export: same score, 26.6 ms per image on 4 CPU cores. See [`reports/results.md`](reports/results.md) and the [model card](MODEL_CARD.md).

## Dataset and license

[RDD2022](https://github.com/sekilab/RoadDamageDetector) (Arya et al., Crowdsensing based Road Damage Detection Challenge 2022): street images from six countries with Pascal VOC boxes. This project uses the labelled images of two countries, Czech Republic (2,829 images, 600 x 600 pixels) and India (7,706 images, 720 x 720 pixels).

**License:** the RoadDamageDetector README states that the images are available under the Creative Commons Attribution ShareAlike 4.0 International license (CC BY-SA 4.0). This repository does not redistribute the images; `make data` reads the country zips from a local folder.

**Download:** the official S3 links in that README answered `AccessDenied` (October 2026), and the figshare copy (record 21431547, one 12 GB archive of all countries) does not resolve from the build environment. The two country zips are therefore taken from the figshare archive by hand. To reproduce: place `RDD2022_Czech.zip` and `RDD2022_India.zip`, or the extracted `Czech` and `India` folders, in `data/raw/`.

## Approach

| Step | What it does |
| --- | --- |
| Data audit | Images, boxes per damage type and country, labels outside the four scored types, box sizes, image sizes, brightness ([`reports/data_audit.md`](reports/data_audit.md)) |
| Split | RDD2022 test images have no labels, so the labelled images are split 70/15/15 per country (7,360 / 1,600 / 1,575 images), by blocks of 25 consecutive images, to keep near duplicate neighbours in one split |
| Model | YOLO11n (2.6 million parameters) from COCO weights; 12 epochs at 512 pixels, batch 16, seed 42 |
| Baseline | Same model with the COCO backbone frozen; only the neck and head learn |
| Augmentation study | Default Ultralytics augmentation against a light setting without mosaic and with smaller scale and color changes |
| Model choice | The run with the highest validation mAP50; the test split is scored only after this choice |
| Error analysis | mAP at IoU 0.5 per class, per country and per lighting bin (brightness tertiles); confusion matrix at confidence 0.25 |
| Deployment | ONNX export; median CPU time per image in PyTorch and in ONNX Runtime; Gradio upload demo |

Training ran on a free Google Colab Tesla T4 GPU with [`notebooks/train_colab.ipynb`](notebooks/train_colab.ipynb), about 17 minutes per run (4 CPU cores need about 3.5 hours per run). Evaluation, export and the report ran on CPU.

## Results on the test split (1,575 images, 1,305 boxes)

| Run | mAP50 | mAP50-95 | AP50 D00 | AP50 D10 | AP50 D20 | AP50 D40 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen backbone (baseline) | 0.214 | 0.087 | 0.177 | 0.065 | 0.394 | 0.221 |
| Full fine tune | 0.261 | 0.108 | 0.245 | 0.102 | 0.457 | 0.242 |
| Full fine tune, light augmentation (main) | **0.288** | **0.121** | 0.272 | 0.089 | 0.512 | 0.279 |

Findings:

* Training every layer adds 0.047 mAP50 over the frozen backbone; light augmentation adds a further 0.027. The study changes mosaic, scale, color and translation together, so it does not show which change matters, and the gap may shrink with more epochs.
* Alligator cracks (D20, large boxes) are detected best; transverse cracks (D10) worst. D10 has 467 boxes in total, 68 of them in India, and India AP50 for D10 is 0.011.
* By country, mAP50 is 0.189 on Czech images and 0.270 on India images. Czech images have fewer boxes and more thin cracks.
* At confidence 0.25 the model rarely confuses one damage type with another, but it misses most damage: 57 of 367 longitudinal cracks and 73 of 528 potholes are found, and no transverse crack. See the [model card](MODEL_CARD.md#errors).
* By lighting, mAP50 varies from 0.275 (medium) to 0.307 (bright); dark images score 0.290. Brightness is not the main source of error.
* All three runs still improved at epoch 12 (see [`reports/figures/learning_curves.png`](reports/figures/learning_curves.png)). A longer schedule is the first next step.

| Runtime (CPU, 4 cores, batch 1, 512 pixels) | Median ms per image | p90 ms |
| --- | ---: | ---: |
| PyTorch | 30.2 | 35.4 |
| ONNX Runtime | 26.6 | 41.5 |

The ONNX model (10.1 MB) scores test mAP50 0.289.

## Reproduce

```
make install
make data SOURCE=<folder with the two zips>
make all
```

`make all` trains on CPU (about 10 hours). For GPU training, upload the two zips and `git archive` of `common/` and this folder to Google Drive and run the Colab notebook; then copy `artifacts/` from its `p06_results.zip` here and run `make evaluate report`. `make smoke` runs the whole pipeline on synthetic zips in about one minute; CI runs it on each push. `make demo` starts the Gradio demo on http://127.0.0.1:7860.

## License of this folder

AGPL-3.0 (see [`LICENSE`](LICENSE)), unlike the MIT license of the rest of the repository. The code uses Ultralytics YOLO, which is AGPL-3.0, and Ultralytics states that software which uses it falls under the same license.
