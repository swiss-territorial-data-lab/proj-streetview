# Detection of manholes on streetview images

<!--Currently do not include the reporjection of the ground truth. To be added later.-->

The scripts in this repo allow detecting manholes on streetview images and projecting the detections to a geographical reference system. Detection with detectron2 and YOLOv11 is available. See below for metrics indicating the result quality.

This project has been done in partnership with the Institut d'Ingénierie du territoire of the HEIG-VD at Yverdon-les-Bains.

**Table of content**

- [Setup](#setup)
    - [Hardware](#hardware)
    - [Software](#software)
    - [Installation](#installation)
- [Data](#data)
- [Workflow](#workflow)
    - [Preprocessing](#preprocessing)
    - [With detectron2](#with-detectron2)
    - [With YOLOv11](#with-YOLOv11)
    - [Postprocessing](#postprocessing)
    - [Reprojection](#reprojection)
    - [Comparison with the pipe cadaster](#comparison-with-the-pipe-cadaster)
- [Additional information](#additional-information)
    - [Project structure](#project-structure)


## Setup

### Hardware

The process was tested on a machine with 16 GB of RAM and a nvidia L4 GPU. At least 32 GB of RAM are recommended.


### Software

TBD

### Installation

The process with detectron2 was run on python 3.8 and the libraries can be installed from `requirements.txt`.<br>
The process with YOLOv11 was run on python 3.10 and the libraries can be installed from `req_YOLOv11.txt`.

**With docker**

TBD

**Without docker**


## Data

The following input data are expected:

* Panoramic images: streetview images with a constant size
* Ground truth (GT): COCO file with the manhole annotations corresponding to the panoramic images
* Annotation validation: JSON-file indicating which GT annotations are valid and which were rejected during control

## Workflow
All the workflow steps with the corresponding command lines are listed below. The user should use either detectron2 or YOLOv11.

### Preprocessing

The preprocessing consists in the following steps:

* Determine image size
* Clip annotations and tiles
    * When clipping images, the pixels corresponding to rejected annoations are masked.

The following command lines are used:

```
python scripts/get_stat_images.py config/config_<DL algo>.yaml
python scripts/prepare_coco_data.py config/config_<DL algo>.yaml
```

The corresponding data paths and parameters are passed through the config file. The following parameters allow to configure the type of task:

```
prepare_coco: <boolean, defaults to False>  # Output COCO files for the detection with detectron2
prepare_YOLOv11: <boolean, defaults to False>  # Output COCO files for the detection with YOLOv11
make_oth_dataset: <boolean, defaults to False>  # Output tiles without annotations on the lower part of the panoramic image.
```

### With detectron2

The training of a model and infrence with detectron2 is done with the following command lines:

```
python scripts/train_detectron2.py config/config_detectron2.yaml
python scripts/infer_with_detectron2.py config/config_detectron2.yaml
```

The results are assessed with the `assess_results.py` script.

```
python scripts/assess_results.py config/config_detectron2.yaml
```

The metrics with the current parameters are given in Table 1.

<i>Table 1: Metrics with detectron2</i>
| dataset | precision | recall | f1 score |
|---------|-----------|--------|----------|
| val     | 0.XXX     | 0.XXX  | 0.XXX     |

### With YOLOv11

Before training YOLOv11, the COCO files must be converted to YOLOv11 format by running `coco_to_YOLOv11.sh`. No configuration is passed explicitly, but it use the parameters in `config/config_YOLOv11.yaml` for the scripts `coco_to_YOLOv11.py` and `redistribute_images.py`.

The training of a model and infrence with YOLOv11 is done with the following command lines:

```
python scripts/train_YOLOv11.py config/config_YOLOv11.yaml
python scripts/infer_with_YOLOv11.py config/config_YOLOv11.yaml
```

The results are assessed with the `assess_results.py` script.

```
python scripts/assess_results.py config/config_YOLOv11.yaml
```

The metrics with the current parameters are given in Table 2.

<i>Table 2: Metrics with YOLOv11</i>
| dataset | precision | recall | f1 score |
|---------|-----------|--------|----------|
| val     | 0.XXX     | 0.XXX  | 0.XXX     |

The optimisation of the hyperparameters is done with the `tune_YOLOv11_w_ray.py` script. The tuning of a model with the `tune` method of YOLOv11 was also tested in the script `tune_YOLOv11_model.py`.

### Postprocessing

The postprocessing consists in reassambling panoramic images from tiles and filtering detections on the score. The following command line is used:

```
python scripts/transform_detections.py config/config_trn_pano.yaml
```

The results can be assessed once the annotations on adjacent tiles are merged for each panoramic image.

```
python scripts/assess_results.py config/config_trn_pano.yaml
```

The metrics with the current parameters are given in Table 3 for detectron2 and Table 4 for YOLOv11.

<i>Table 3: Metrics with postprocessing for detectron2</i>
| dataset | precision | recall | f1 score |
|---------|-----------|--------|----------|
| val     | 0.XXX     | 0.XXX  | 0.XXX     |

<i>Table 4: Metrics with postprocessing for YOLOv11</i>
| dataset | precision | recall | f1 score |
|---------|-----------|--------|----------|
| val     | 0.XXX     | 0.XXX  | 0.XXX     |

### Reprojection


### Comparison with the pipe cadaster

The final detections are compared with the existing layer of the pipe cadaster. The precision and recall are calulated with the pipe cadaster considered as ground truth. Areas with discrepencies area highlighted.

TBD

_Warning_: No example dataset is provided here for the pipe cadaster.

## Additional information


### Project structure