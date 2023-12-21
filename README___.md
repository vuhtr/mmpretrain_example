# Text and Wire Classification

## Installation (if neccesary)

```
conda create -n classify python=3.9 -y
conda activate classify
bash scripts/install.sh
```

## Text Classification

- Best model with MobileNetV2 backbone (including `.pth` and `.py` for pytorch, `.onnx` for ONNX):

```bash
gsutil -m cp -r \
  "gs://snapedit-chuonghm/vu_models/text_classifer" \
  .
```

Deployment and inference scripts: Check the folder `deploy`
