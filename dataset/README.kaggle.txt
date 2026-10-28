Supplemental Kaggle PPE data attribution

Dataset: Safety-PPE Detection Unified Dataset v1 (stratified re-split)
Kaggle page: https://www.kaggle.com/datasets/quocdatlam/ppe-unified-v1
Kaggle owner: Quoc Dat Lam
Kaggle dataset version: 1
License: Creative Commons Attribution 4.0 International (CC BY 4.0)
License text: https://creativecommons.org/licenses/by/4.0/

The source dataset contains 17,951 images and 51,179 boxes in its original
train/val/test splits. Its card attributes the merged source datasets below
to their Roboflow publishers; those source datasets are stated to be CC BY 4.0:

- Hard Hat Detection v2 — computer-vision-filqz
- PPE detection v1 — testcasque
- PPE Detection v5 — mohamed-traore-2ekkp

The source's positive object classes were mapped to this project's IDs:
helmet (0) -> Helmet (4), vest (2) -> Vest (7), gloves (4) -> Glove (3),
goggles (6) -> Glass (2), boots (8) -> Boots (0), and mask (10) -> Mask (5).
The source's no_helmet, no_vest, no_gloves, no_goggles, no_boots, and no_mask
classes were omitted because this project's detector models physical PPE
objects, not missing-equipment labels. The source has no Person or
Ear-protection class. Source images without any retained positive PPE box and
exact image-content duplicates already present in the project were excluded.
Source train/val/test split membership was retained.

The merge added 12,323 images: 8,604 train, 2,520 validation, and 1,199 test.
It excluded 5,308 source images without a retained positive PPE box and 320
exact image duplicates of images already in the project. The combined project
dataset now contains 12,974 train, 3,100 validation, and 1,613 test images.

This project dataset is modified and combines data from multiple sources.
Retain this attribution and the CC BY 4.0 license when redistributing the
included source data or adaptations.
