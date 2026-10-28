# AI Workplace Safety Monitoring

An experimental computer-vision project for detecting workplace personal
protective equipment (PPE), associating PPE detections with people, and
recording potential violations.

## Features

- YOLO-based detection for Boots, Ear-protection, Glass, Glove, Helmet, Mask,
  Person, and Vest.
- PPE association and configurable required-equipment rules.
- Webcam monitoring and a Flask dashboard.
- Live monitoring can be started, stopped, or restarted from the dashboard;
  leaving the Live tab stops the camera and releases it.
- SQLite storage for violation records and evidence images.
- Dataset integrity checking and model training scripts.

## Dataset

The merged YOLO dataset is in `dataset/`, with train, validation, and test
splits (12,974 / 3,100 / 1,613 images). The class IDs are defined in
`dataset/data.yaml`; Glove is class `3`.
Dataset structure and annotation guidance are documented in
[`PPE_DATASET.md`](PPE_DATASET.md).
The original Roboflow dataset attribution and its CC BY 4.0 license are
preserved in `dataset/README.dataset.txt`. Added PPE_Dectection v4 data and
its class mapping are attributed in `dataset/README.supplement.txt`. The
Kaggle PPE supplement is attributed in `dataset/README.kaggle.txt`; see
[`PPE_DATASET.md`](PPE_DATASET.md) for combined dataset details.

Validate the dataset from the project root:

```powershell
python check_dataset.py
```

## Setup

Use Python 3.10 or a compatible version supported by the installed
Ultralytics/PyTorch packages. On Windows:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Train

The training script fine-tunes the latest `ppe_safety_model_v*` checkpoint when
one exists; otherwise it starts from `yolo11n.pt`. Run outputs are written
under `runs/`:

```powershell
python train.py
```

The pretrained starting weights and generated run outputs are not stored in
this repository. Obtain model weights only from sources whose license permits
your intended use.

## Run

Place a compatible trained checkpoint at the path expected by the selected
script, then run it from the project root:

```powershell
python detect.py
python webcam.py
python app.py
```

The dashboard selects the newest `ppe_safety_model_v*` checkpoint. The webcam
and standalone detection scripts still target the V3 run path. Camera
performance depends on lighting, camera placement, and detection quality.
Confirm detections manually before relying on alerts for safety decisions.

## Project layout

```text
app.py
database.py
detect.py
webcam.py
safety_rules.py
train.py
check_dataset.py
dataset/
templates/
tools/
```

Generated weights, the local SQLite database, evidence images, archives, and
Python virtual environments are intentionally excluded from version control.
