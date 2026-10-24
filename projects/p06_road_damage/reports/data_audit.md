# Data audit: RDD2022, Czech and India

Labelled images only (the `train` folders of RDD2022; its `test` folders have no labels). Split by blocks of 25 consecutive file names within each country, 70/15/15, seed 42.

## Images per country and split

| country   |   train |   val |   test |   total |
|:----------|--------:|------:|-------:|--------:|
| Czech     |    1979 |   425 |    425 |    2829 |
| India     |    5381 |  1175 |   1150 |    7706 |

Share of images without any of the four scored damage types: Czech 62.1%, India 58.2%.

Image sizes in pixels (width, height): Czech: 600 x 600; India: 720 x 720. Annotation files whose stated size differs from the image: 0.

## Boxes per label

| label   | damage type         |   Czech |   India |   total |
|:--------|:--------------------|--------:|--------:|--------:|
| D00     | longitudinal crack  |     988 |    1555 |    2543 |
| D10     | transverse crack    |     399 |      68 |     467 |
| D20     | alligator crack     |     161 |    2021 |    2182 |
| D40     | pothole             |     197 |    3187 |    3384 |
| D01     | not scored, dropped |       0 |     179 |     179 |
| D0w0    | not scored, dropped |       0 |       1 |       1 |
| D11     | not scored, dropped |       0 |      45 |      45 |
| D43     | not scored, dropped |       0 |      57 |      57 |
| D44     | not scored, dropped |       0 |    1062 |    1062 |
| D50     | not scored, dropped |       0 |      28 |      28 |

## Box size, scored labels

| label   |   boxes |   median share of image area |   largest share |
|:--------|--------:|-----------------------------:|----------------:|
| D00     |    2543 |                        0.014 |           0.252 |
| D10     |     467 |                        0.011 |           0.117 |
| D20     |    2182 |                        0.09  |           0.668 |
| D40     |    3384 |                        0.008 |           0.387 |

## Lighting bins (tertiles of mean luminance, 0 to 255)

| lighting   |   min |   max |   count |
|:-----------|------:|------:|--------:|
| bright     | 136.4 | 188.9 |    3512 |
| dark       |  35.1 | 121.6 |    3512 |
| medium     | 121.6 | 136.4 |    3511 |
