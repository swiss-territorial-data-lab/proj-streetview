# Detection of manholes on street view images

The scripts in this repo allow detecting manholes on street view images and projecting the detections to a geographical reference system. Detection with detectron2 and YOLO is available. Detailed method and results are discussed on the [STDL tech website](https://www.tech.stdl.ch/PROJ-STREETVIEW)

This project has been done in partnership with the Institut d'Ingénierie du territoire of the HEIG-VD at Yverdon-les-Bains.

**Table of content**

- [Setup](#setup)
    - [Installation](#installation)
- [Data](#data)
- [Workflow](#workflow)
    - [Dataset preparation](#dataset-preparation)
    - [Deep learning](#deep-learning)
        - [Preprocessing](#preprocessing)
        - [With detectron2](#with-detectron2)
        - [With YOLO](#with-yolo)
        - [Postprocessing](#postprocessing)
    - [Geo-localization](#geo-localization)
    - [Comparison with the pipe cadaster](#comparison-with-the-pipe-cadaster)
- [Additional information](#additional-information)
    - [Project structure](#project-structure)


## Setup

The process was tested on a machine with Ubuntu 22.04, 32 GB of RAM and a nvidia L4 GPU with 16 GB of VRAM.
Please note that detectron2 can only be run on Linux-based systems or macOS.

### Installation

The process with YOLOv11 was run on python 3.10 and the libraries can be installed from `req_yolo.txt`.<br>
The process with detectron2 was run on python 3.8 and the libraries can be installed from `requirements.txt`.

> [!NOTE]  
> The results presented in the official documentation were obtained with a batch size of 25 images for the YOLO training. A batch size of 10 images is set in the default parameters of this repo, because the use of docker 

**Without docker**

To use YOLOv11, python 3.10 is expected. To use detectron2, python 3.8 is required.

All libraries can be installed with `pip install -r req_yolo.txt` for YOLO and `pip install -r config/detectron2/req_det2.txt` for detectron2.

**With docker**

A docker image provided by ultralytics and completed for this project is available in this repo. Ensure you have the nvidia-container-toolkit installed first.

## Data

The following input data are expected for the _deep-learning part_:

* Panoramic images: street view images with a constant size;
* Ground truth (GT) annotations on street view images: COCO file with the manhole annotations corresponding to the panoramic images;
* Validated annotations: COCO file with only the manhole annotations that were validated by visual control.

The following input data are expected for the part about _geo-localization_:

* Trajectory information: table withtime, position and orientation of the camera at each image;
* GT vector layer: georeferenced vector file with the expected geolocalized manholes.

The following data are expected for the part about _cadaster control_:

* Area of interest: georeferenced vector file with either the AOI as polygons or the position of the camera at each image;
* Manholes: georeferenced vector file with the manholes from the pipe cadaster as points or as polygons.

Example data can be found in the `data/RCNE` folder to test the workflow. The images need to be downloaded with the `get_images_rcne.py` script.

```
python scripts/utils/get_images_rcne.py config/config_yolo.yaml
```

## Workflow

### Dataset Preparation

This project uses ortho-intensity rasters derived from LiDAR data to semi-automatically generate annotations for deep learning. This step is optional as ground truth could be generated in another way and no ground truth is needed for inference.

The employed method is the following:
- Ground points are first extracted using the Cloth Simulation Filter (CSF) to remove above-ground objects that could occlude manholes. 
- The remaining data is interpolated with inverse-distance weighting to create intensity and elevation rasters. 
- Circular features are then detected using the Hough transform, and combined with elevation data to regress each candidate to a flat circle in the local 3D coordinate system. 
- Finally, validated 3D targets are projected onto street-view images using camera metadata. 

See [dataset preparation](./dataset_preparation/README.md) for detailed instructions.

### Deep learning

All the deep-learning steps with the corresponding command lines are listed below. The user should use either detectron2 or YOLO. <br>
Input files and parameters are passed through config files. Recurring parameters are defined in `scripts/utils/constants.py`. Path for the detectron2, YOLO, and "COCO for yolo conversion" folder can be indicated in with `<DETECTRON2_FOLDER>`, `<YOLO_FOLDER>` and `<COCO_FOR_YOLO_FOLDER>` respectively in the config files. This string are then automatically replaced by the corresponding path in the script.

#### Preprocessing

The preprocessing consists in the following steps:

* Determine image size
* Clip annotations and tiles
    * When clipping images, the pixels corresponding to rejected annoations are masked.
    * If no valid annotations are passed, the only the inference dataset is prepared without annotations and without masked objects.

The data paths and parameters are passed through the config file. The following parameters allow to configure the type of task:

```
taks:
    test_only: <boolean, defaults to False> # Output all the annotations and corresponding tiles in the test dataset.
    make_oth_dataset: <boolean, defaults to False>  # Output tiles without annotations on the lower part of the panoramic image.
    coco:
        prepare_data: <boolean, defaults to False>  # Output the COCO files and images for detectron2.
        subfolder: <string>  # Subfolder name for the COCO files.
    yolo:
        prepare_data: <boolean, defaults to False>  # Output the COCO files and images for YOLO.
        subfolder: <string>  # Subfolder name for the YOLO files.
```

In case data are produced for both coco and yolo, hard links are created to limit the amount of image data.

The following command lines are used:

```
python scripts/deep-learning/get_stat_images.py config/config_<DL algo>.yaml
python scripts/deep-learning/prepare_coco_data.py config/config_<DL algo>.yaml
```

#### With detectron2

The training of a model and inference with detectron2 is done with the following command lines:

```
python scripts/deep-learning/detectron2/train_detectron2.py config/detectron2/config_detectron2.yaml
python scripts/deep-learning/detectron2/infer_with_detectron2.py config/detectron2/config_detectron2.yaml
```

The results are assessed with the `assess_results.py` script.

```
python scripts/deep-learning/assess_results.py config/config_detectron2.yaml
```

After the manual search for the hyperparameters, the best models for the various AOI tested achieved around 88% precision and 75% recall.

#### With YOLO

Before training YOLO, the COCO files must be converted to yolo format by running `coco_to_yolo.sh`. No configuration is passed explicitly, but it use the parameters in `config/config_yolo.yaml` for the scripts `coco_to_yolo.py` and `redistribute_images.py`. A path is given directly in the bash script to remove the images once used.

> [!WARNING]  
> An exception might be raised because of the hard-coded path for YOLO datasets. In that case, please follow the instructions of the execption message.  
> The risk is that the path to the datasets is not mounted in the docker container. If so, `coco_to_yolo.sh` will have to be run every time the container is started.

Before training, it is possible to optimize the hyperparameters with the `tune_yolo_model.py` script or the `tune_yolo_w_ray.py` script.

* `tune_yolo_model.py`: use the built-in method of YOLO. Only allow to optimize the float hyperparameters.
* `tune_yolo_w_ray.py`: use the `ray` library to optimize the hyperparameters. Allow to optimize any hyperparameter.

The training of a model and inference with YOLO are done with the following command lines:

```
bash scripts/deep-learning/yolo/coco_to_yolo.sh
python scripts/deep-learning/yolo/train_yolo.py config/config_yolo.yaml
python scripts/deep-learning/yolo/validate_yolo.py config/config_yolo.yaml
python scripts/deep-learning/yolo/infer_with_yolo.py config/config_yolo.yaml
```

The results are assessed with the `assess_results.py` script.

```
python scripts/deep-learning/assess_results.py config/config_yolo.yaml
```

After optimization of the hyperparameters, the best models for the various AOI tested achieved around 93% precision and 92% recall.

**Hyperparameter optimization**

The optimization of the hyperparameters is done with the `tune_yolo_w_ray.py` script. The tuning of a model with the `tune` method of YOLOv11 was also tested in the script `tune_yolo_model.py`.

#### Postprocessing

The postprocessing consists in reassembling panoramic images from tiles and filtering detections on the score. The following command line is used:

```
python scripts/deep-learning/transform_detections.py config/config_transfo_pano.yaml
```

The results can be assessed once the annotations on adjacent tiles are merged for each panoramic image.

```
python scripts/deep-learning/clipped_labels_to_panoramic.py config/config_transfo_pano.yaml
python scripts/deep-learning/assess_results.py config/config_transfo_pano.yaml
```

The impact of this post-processing on the metrics is negligible.

### Geo-localization   

1. Inspect camera model and image type. Calibrate the camera trajectory if necessary. For constant offsets of camera postion or orientation, you can fix them when defining projection function. 

2. Run the main workflows:
   - [`geo_localization_cubemap_pano.ipynb`](./scripts/geo_localization/geo_localization_cubemap_pano.ipynb) for cubemap-based panoramas.
   - [`geo_localization_spherical_pano.ipynb`](./scripts/geo_localization/geo_localization_spherical_pano.ipynb) for spherical/equirectangular panoramas.
   - Detailed instruction on input, output and parameters to finetune can be found in notebook markdown cell.

3. Final geo-localized detections are saved in geopackage file. Qualitative evaluation can be conducted with ortho-intensity if no ground truth available.  

### Comparison with the pipe cadaster

The final detections are compared with the existing layer of the pipe cadaster. The precision and recall are calculated with the pipe cadaster considered as ground truth. Areas with discrepancies are highlighted.

```
python scripts/cadaster_control.py config/config_cadaster.yaml
```


## Additional information


### Project structure

The project is structured as follows:

```
.
├── README.md
├── LICENSE
├── Dockerfile
├── docker-compose.yml
├── req_yolo.in
├── req_yolo.txt
├── config
│   ├── detectron2      # Configuration files and required libraries for detectron2
│   ├── *.yaml          # Configuration files for YOLO
├── data/RCNE           # Example data to test the workflow
└── scripts
    ├── deep-learning
    │   ├── detectron2  # Scripts to train and infer with detectron2
    │   ├── yolo        # Scripts to train and infer with YOLO
    │   ├── *.py        # Scripts for preprocessing, postprocessing, and assessment of the data in the deep-learning workflow
    ├── utils           # Utility scripts
    ├── cadaster_control.py # Script to compare the results with the pipe cadaster
    └── get_images_rcne.py  # Script to download the example images
```
