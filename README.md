# MalVision 🔴🟢

**MalVision** is a malware detection tool that classifies files by converting them into grayscale images and analyzing their binary structure with a deep learning model.

## How it works

1. **Binary to image** — the uploaded file is read byte by byte and mapped into a grayscale image, where each pixel represents a byte value (0–255).
2. **Classification** — the image is fed into a fine-tuned ResNet-50 model trained on the [MalImg dataset](https://www.kaggle.com/datasets/keerthicheepurupalli/malimg-dataset-9339-images), which contains 25 malware families.
3. **Result** — the model returns the predicted class and a confidence score. If the file does not match any known malware pattern, it is classified as **benign**.

## Model performance

Evaluated on the MalImg test set:

| Metric | Score |
|---|---|
| Accuracy | **97.23%** |
| Macro F1 | 0.93 |
| Weighted F1 | 0.97 |

The model was trained on 25 malware families from MalImg plus a benign class built from macOS system binaries, for a total of 26 classes.

## Supported malware families

Adialer.C, Agent.FYI, Allaple.A, Allaple.L, Alueron.gen!J, Autorun.K, C2LOP.P, C2LOP.gen!g, Dialplatform.B, Dontovo.A, Fakerean, Instantaccess, Lolyda.AA1, Lolyda.AA2, Lolyda.AA3, Lolyda.AT, Malex.gen!J, Obfuscator.AD, Rbot!gen, Skintrim.N, Swizzor.gen!E, Swizzor.gen!I, VB.AT, Wintrim.BX, Yuner.A

## Tech stack

- **Model** — ResNet-50 (transfer learning, fine-tuned on MalImg)
- **Framework** — PyTorch + torchvision
- **UI** — Streamlit
- **Image processing** — Pillow, NumPy

## Installation

```bash
git clone https://github.com/Baffo2906/MalVision.git
cd MalVision
pip install -r requirements.txt
```

## Usage

```bash
streamlit run app.py
```

Upload any file, check the box if it is already a binary-converted PNG image, and click **Analyze**.

## Project structure

```
MalVision/
├── app.py                        # Streamlit UI
├── model/
│   └── classifier.py             # Inference
├── pipeline/
│   ├── binary_to_image.py        # Binary → grayscale image
│   ├── dataset_loader.py         # Dataset loading
│   ├── train.py                  # Model training
│   └── evaluate.py               # Model evaluation
└── .streamlit/
    └── config.toml               # UI theme
```

## Reference

This project is inspired by:
> Nataraj, L., et al. (2011). *Malware Images: Visualization and Automatic Classification*. VizSec '11.

---

*by Raffaele Passaro*
