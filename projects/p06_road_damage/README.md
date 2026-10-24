# Project 6: Road damage detection

Detect and classify road surface damage (longitudinal, transverse and alligator cracks, and potholes) in street images, so that a city can rank repairs.

**Status: in progress.** The pipeline is built and tested on synthetic data; training on RDD2022 is pending.

## Dataset and license

[RDD2022](https://github.com/sekilab/RoadDamageDetector) (Arya et al., Crowdsensing based Road Damage Detection Challenge 2022): street images from six countries with Pascal VOC boxes. This project uses the labelled images of two countries, Czech Republic and India.

**License:** the RoadDamageDetector README states that the images are available under the Creative Commons Attribution ShareAlike 4.0 International license (CC BY-SA 4.0). This repository does not redistribute the images; `make data` reads the country zips from a local folder.

**Download:** the official S3 links in that README answered `AccessDenied` (October 2026), and the figshare copy (record 21431547, one 12 GB archive of all countries) does not resolve from the build environment. The two country zips are therefore taken from the figshare archive by hand. To reproduce: place `RDD2022_Czech.zip` and `RDD2022_India.zip`, or the extracted `Czech` and `India` folders, in `data/raw/`.

## Approach

| Step | What it does |
| --- | --- |
| Data audit | Images, boxes per damage type and country, labels outside the four scored types, box sizes, image sizes, brightness |
| Split | RDD2022 test images have no labels, so the labelled images are split 70/15/15 per country, by blocks of 25 consecutive images, to keep near duplicate neighbours in one split |
| Model | YOLO11n (2.6 million parameters) from COCO weights, trained on 4 CPU cores |
| Baseline | Same model with the COCO backbone frozen; only the neck and head learn |
| Augmentation study | Default Ultralytics augmentation against a light setting without mosaic |
| Error analysis | mAP at IoU 0.5 per class, per country and per lighting bin (brightness tertiles); confusion matrix |
| Deployment | ONNX export; median CPU time per image in PyTorch and in ONNX Runtime; Gradio upload demo |

## Reproduce

```
make install
make data SOURCE=<folder with the two zips>
make all
```

`make smoke` runs the whole pipeline on synthetic zips in about one minute; CI runs it on each push. `make demo` starts the Gradio demo on http://127.0.0.1:7860.

## License of this folder

AGPL-3.0 (see [`LICENSE`](LICENSE)), unlike the MIT license of the rest of the repository. The code uses Ultralytics YOLO, which is AGPL-3.0, and Ultralytics states that software which uses it falls under the same license.
