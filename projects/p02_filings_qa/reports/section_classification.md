Pages: 5,418 from 62 10-K filings; test companies: 8 of 30.

| model                                   |   macro F1 |   accuracy |
|:----------------------------------------|-----------:|-----------:|
| majority class                          |      0.102 |      0.441 |
| TF-IDF + logistic regression            |      0.625 |      0.78  |
| MiniLM embeddings + logistic regression |      0.598 |      0.655 |

Pages per section:

| section               |   test |   train |
|:----------------------|-------:|--------:|
| business              |    161 |     514 |
| financial statements  |    522 |    2049 |
| management discussion |    195 |    1036 |
| market risk           |     11 |      27 |
| other                 |     92 |     103 |
| risk factors          |    203 |     505 |

Per class, TF-IDF + logistic regression:

```
                       precision    recall  f1-score   support

             business      0.869     0.783     0.824       161
 financial statements      0.835     0.941     0.885       522
management discussion      0.608     0.749     0.671       195
          market risk      1.000     0.273     0.429        11
                other      0.478     0.120     0.191        92
         risk factors      0.789     0.719     0.753       203

             accuracy                          0.780      1184
            macro avg      0.763     0.597     0.625      1184
         weighted avg      0.768     0.780     0.760      1184
```
