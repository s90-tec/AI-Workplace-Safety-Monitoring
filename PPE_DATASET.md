# PPE Dataset Guide

YOLO (You Only Look Once) detects objects by predicting bounding boxes and
class labels in a single pass over an image. The application adds workplace
safety rules to those detections, such as deciding that a person without a
helmet is a violation.

## Project classes

The project configuration defines these eight class IDs in `dataset/data.yaml`:

```text
0 Boots
1 Ear-protection
2 Glass
3 Glove
4 Helmet
5 Mask
6 Person
7 Vest
```

These names match the original Roboflow annotations. The original metadata is
preserved as `dataset/roboflow-export.yaml`, and the safety-violation classes
`no_helmet`, `no_vest`, and `no_gloves` are not part of this dataset.

The training, validation, and test splits also include a compatible subset of
Roboflow's `PPE_Dectection` v4 export. Its source, CC BY 4.0 attribution, class
mapping, and omitted negative-equipment tags are documented in
`dataset/README.supplement.txt`. The added export contains labels for Boots,
Gloves, Goggles, Hardhat, Person, and Safety Vest; Goggles are mapped to the
project's Glass class. It does not add Ear-protection or Mask examples, so
those classes remain underrepresented.

The dataset also includes compatible positive PPE labels from Kaggle's
`ppe-unified-v1` (CC BY 4.0). Its source attribution and class mapping are
preserved in `dataset/README.kaggle.txt`. The source adds Helmet, Vest, Gloves,
Goggles, Boots, and Mask; negative-compliance labels are omitted because this
project detects physical PPE objects rather than "missing PPE" objects.
Ear-protection remains unavailable from this source. After filtering samples
without retained PPE boxes and exact duplicates, the combined dataset contains
12,974 train, 3,100 validation, and 1,613 test images.

## Directory layout

Put source images and matching YOLO label files here:

```text
dataset/
  data.yaml
  images/
    train/
    val/
    test/
  labels/
    train/
    val/
    test/
```

The current export has been flattened into this layout. Every split currently
has matching image and label filenames.

Every image should have a label file with the same filename. Each label line
uses normalized YOLO coordinates:

```text
class_id center_x center_y width height
```

All coordinate values must be between 0 and 1. Draw boxes around the complete
visible object: the worker for `person`, the head protection for `helmet`, and
the garment for `vest`.

## Helmet violation policy

There are two valid approaches. If the dataset contains explicit examples of
missing equipment, annotate them as `no_helmet` or `no_vest`. Otherwise, use
the model to detect `person` and `helmet`, then apply application logic:

```text
person detected
+ no helmet near the person's head
= helmet violation
```

The second approach avoids treating an invisible object as a physical object,
but it requires a consistent rule for matching helmets to people.

## Data sources and annotation

Use images from construction sites, factories, warehouses, industrial areas,
or road construction. Public datasets are acceptable only when their license
allows the intended research, modification, and publication. Do not assume
that an image found through a search engine is available for reuse.

Annotate with a tool such as Roboflow or LabelImg, then export in YOLO format.
Keep training and validation images separate, and make sure both sets contain
representative lighting, camera angles, clothing, and working environments.

## Training command

After adding annotated images and labels, train a first model with:

```powershell
python train.py
```