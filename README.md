# METASHAPE & DETECTRON FOR SCALLOPS

## Tools for:
* Metashape reconstruction from Gopro video and ROV data
* Projecting metashape polygon annotations to camera images
* Rendering novel viewpoints from metashape reconstructions
* Training MaskRCNN using Facebook Detectron2
* Running and Analysing experiments on various augmentations and datasets

## Installation

Install requirements:
```
pip install -r requirements.txt
```
Install Detectron2 from source:
```
git clone git@github.com:facebookresearch/detectron2.git
cd detectron2
pip install .
```
