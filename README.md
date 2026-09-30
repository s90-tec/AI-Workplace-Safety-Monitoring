# AI Workplace Safety Monitoring

An experimental computer-vision project for detecting workplace personal
protective equipment (PPE), associating PPE detections with people, and
recording potential violations.

## Features

- YOLO-based detection for Boots, Ear-protection, Glass, Glove, Helmet, Mask,
  Person, and Vest.
- PPE association and configurable required-equipment rules.
- Webcam monitoring and a Flask dashboard.
- SQLite storage for violation records and evidence images.
- Dataset integrity checking and model training scripts.

## Dataset

The merged YOLO dataset is in `dataset/`, with train, validation, and test
splits. The class IDs are defined in `dataset/data.yaml`; Glove is class `3`.
Dataset structure and annotation guidance are documented in
[`PPE_DATASET.md`](PPE_DATASET.md).
The original Roboflow dataset attribution and its CC BY 4.0 license are
preserved in `dataset/README.dataset.txt`.

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

The training script uses `yolo11n.pt` as its pretrained starting point and
writes run outputs under `runs/`:

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

The webcam and detection scripts currently target the V3 run path. V3 training
and evaluation are in progress; the V3 checkpoint is not included here and
should not be treated as a final validated model. Camera performance depends
on lighting, camera placement, and detection quality. Confirm detections
manually before relying on alerts for safety decisions.

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
