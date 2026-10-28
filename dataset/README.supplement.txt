Supplemental PPE data attribution

Source: PPE_Dectection, version 4, provided by a Roboflow user
Project: https://universe.roboflow.com/himanshu-bharati/ppe_dectection-dtt4q
Export: https://universe.roboflow.com/himanshu-bharati/ppe_dectection-dtt4q/dataset/4
License: Creative Commons Attribution 4.0 International (CC BY 4.0)
License text: https://creativecommons.org/licenses/by/4.0/
Dataset card: https://huggingface.co/datasets/HB1204/PPE_Detection

The downloaded source comprised 3,086 images in train, valid, and test splits.
Its labels were converted from the source YOLOv8 class IDs to this project's
eight-class IDs. Source Boots, Gloves, Goggles, Hardhat, Person, and Safety
Vest were retained; Goggles map to this project's Glass class. The source
NO-Gloves, NO-Goggles, NO-Hardhat, NO-Safety Vest, and None labels were
omitted because they are not physical object classes in this project. The
source did not label Ear-protection or Mask.

Images were added to the corresponding project split. Exact duplicate image
content already present in the project or another source split was skipped.
This is a modified dataset; source split boundaries were retained for the
included classes, and exact duplicate annotations within a label file were
removed.
