# DL_T05_1B: Paddy leaf disease detection for a low-cost phone

Team 5 | Task 1 | Variant 1B (Paddy Doctor, Kaggle) | CM701PC / CM751PC Deep Learning, MGIT

A MobileNetV2 (width 0.35) classifier for 10 paddy classes (9 diseases + normal), exported to TensorFlow Lite for on-device use.

## Status (read this first)
- Numbers below come from the team's own Google Colab runs (one seed per run unless stated).
- `src/` modules were smoke-tested on random data. The full notebook has to be run once by the team before claiming it reproduces these numbers.
- Retraining gives similar but **not identical** numbers (GPU non-determinism, different random streams).
- **Field photos:** real field photographs could not be collected within the available time and resources. A *simulated field-shift proxy* (code-generated degradations on held-out images) is used instead. It is not a substitute for real field photos.

## Folder layout
```
DL_T05_1B/
  notebook/DL_T05_1B.ipynb   run top to bottom in Colab (T4 GPU)
  src/                       config, data, models, augment, shift, train, evaluate
  splits/                    train/val/test CSVs (authoritative split), classes.json, rare_classes.json
  models/                    E7.keras, paddy_E7_float32.tflite, paddy_E7_dynamic.tflite, E4.keras, mnet035_E3.keras
  results/                   copy of outputs/ (CSVs, confusion matrices, figures)
  docs/                      viva_sheet.md, ai_use_appendix.md
  requirements.txt
```
The notebook expects the project at `MyDrive/DL_T05_1B/` (inside it, `src/`, `splits/`, `models/`, `outputs/`).
The image cache (`cache/`, about 1.5 GB) is rebuilt automatically and is not part of the ZIP.

## Setup
1. Kaggle: join the competition "Paddy Doctor: Paddy Disease Classification" (accept the rules) and create an API token.
2. Upload this folder to Google Drive as `MyDrive/DL_T05_1B/`.
3. Open `notebook/DL_T05_1B.ipynb` in Colab, select a T4 GPU, run all cells. Default `RUN_TRAINING = False` loads saved models.

## Method summary
- Data: 10,407 labelled images (480x640), 10 classes. Two rare classes = two smallest: `bacterial_panicle_blight` (337), `bacterial_leaf_streak` (380).
- Split: stratified 70/15/15, seed 42. MD5 check found 4 exact duplicates inside test (kept), 12 train/val images identical to test and 22 train images identical to val (removed). Final sizes 7252 / 1559 / 1562. Near-duplicates cannot be detected this way.
- Images resized once to 224x224 (the same resize must be used on the phone). Rescaling is inside the model.
- All design choices judged on validation. Test split used once, for the final model.
- Class weights ("balanced") in the loss to protect rare classes.

## Experiments (validation, one seed each)
| ID | Setup | Val macro F1 | Recall blight / streak |
|---|---|---|---|
| E1-old (x2) | scratch CNN, 99,690 params, unseeded, provenance not recorded | 0.366 and 0.668 | 0.60/0.702 and 0.88/0.930 |
| E1 seeds 1 and 2 | scratch CNN, explicit seeds (a third seed was not completed in time) | 0.570 and 0.735 | 0.44/0.737 and 0.82/0.772 |
| E2 | MobileNetV2-0.35, frozen head, augmentation (E3 recipe) | 0.509 | 0.60 / 0.719 |
| E3 | + fine-tuning, 15 epochs | 0.826 | 0.90 / 0.965 |
| E4 | E3 without augmentation | 0.890 | 0.88 / 0.930 |
| E5 | E4 without class weights | 0.895 | 0.76 / 0.930 |
| E6 | E4 with width 0.5 (719k params) | 0.886 | 0.84 / 0.860 |
| E7 | in-model augmentation incl. blur/noise/low-res, 30 epochs, patience 6 | **0.909** | 0.92 / 0.912 |

E7 differs from E4 in augmentation, epoch cap and patience; their effects were not isolated.

## Final results: E7 on the TEST split
| Target (PDF) | Required | Result | Met |
|---|---|---|---|
| TFLite size | < 10 MB | 1.66 MB float32 / 0.58 MB dynamic int8 | yes |
| Latency, 1 CPU thread | < 300 ms | 2.6 ms / 4.7 ms median (Colab CPU, model only) | yes (proxy) |
| Macro F1 | >= 0.80 | 0.889 (TFLite float32 0.8888, int8 0.8832) | yes |
| Recall `bacterial_panicle_blight` | >= 0.75 | 0.882 (45/51) | yes |
| Recall `bacterial_leaf_streak` | >= 0.75 | 0.912 (52/57) | yes |

Each rare class has only 51-57 test images, so one image moves recall by about 2 points.

## Field-shift proxy (simulated, NOT real field photos), E7 on test images
| Data | Macro F1 | Recall blight | Recall streak |
|---|---|---|---|
| clean | 0.889 | 0.882 | 0.912 |
| shift A (in-family) | 0.742 | 0.667 | 0.807 |
| shift B (different types) | 0.611 | 0.569 | 0.737 |

Panicle blight recall is below 0.75 on both proxies (the PDF sets no numeric target for field images).

## Limitations
Single seed per experiment (E1 scratch CNN: four runs, macro F1 0.366 to 0.735, so variance is large); proxy shift is code-made and only low light is unrelated to E7's training degradations; latency is a Colab CPU proxy (excludes JPEG decoding/resizing); exact-duplicate check cannot catch near-duplicates; E1 old rows have no recorded provenance.

## Files to submit (naming `DL_T05_1B`)
Report PDF `DL_T05_1B.pdf`, this project as `DL_T05_1B.zip`, the `.tflite` file, confusion matrices and per-class tables (in `results/`).
