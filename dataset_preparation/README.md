# Dataset preparation instruction

This folder contains Jupyter notebooks and GUI utilities for preparing data, generating ground-truth labels, and validating datasets for object detection models.

## Contents

- **3d_gt_to_cubemap.ipynb**: Converts 3D ground-truth annotations into per-face cubemap labels.
- **3d_gt_to_spherical.ipynb**: Projects 3D ground-truth annotations into spherical image labels.
- **lidar_to_3d_gt.ipynb**: Builds 3D ground-truth from LiDAR inputs.
- **COCO_validator.py**: Lightweight sanity checker for COCO-style datasets (second step validation to remove occluded objects in street-view images).

## Environment setup

For package management, we strongly suggest using Conda as it is the easiest way for PDAL installation across different OS.

Conda:

```bash
cd ~/proj-streetview/dataset_preparation
conda create -n data_prep python=3.10 -y
conda activate data_prep

conda install -c conda-forge pdal python-pdal
pip install -r requirements.txt
```

Launch Jupyter (choose either Lab or classic) after activating your environment:

```bash
jupyter lab
# or
jupyter notebook
```

## Data prerequisites

- Mobile mapping LiDAR acquisition
- Street-view images
- Image trajectory 
- Camera model and metadata 


   *`Note:`*

   *`1. All annotations are regularized in COCO-format.`*

   *`2. All coordinate reference systems (CRS) expected by these notebooks are 'EPSG:2056'. Please modify both notebooks and 'scripts/utils/' utilities for other CRS.`*


## Usage notes

Run the notebooks to generate and validate ground-truth:

1. Use `lidar_to_3d_gt.ipynb` to derive 3D Ground Truth (GT) from LiDAR if needed.
2. **First validation**: Check initial proposed manhole geometry with QGIS. Visualize both vector polygons, camera trajectory and generated LiDAR intensity layer. For ambiguous targets, choose one nearby camera, open corresponding image and validate if the target exists.  
3. Convert 3D GT to image-space labels using `3d_gt_to_cubemap.ipynb` or `3d_gt_to_spherical.ipynb` depending on your panorama format.
4. **Second validation**: When projecting 3D GT to street-view images, targets are sometimes occluded by above-ground static objects (parking vehicles/garbage/soil) or other moving ones (vehicles/pedestrians). To ensure the rigour of image ground truth. It is necessary to remove them. We provide a simple script with GUI for manual check:
  
    ```bash
    python COCO_validator.py --ann path/to/annotations.json --images path/to/images_dir   
    ```



### Tips and troubleshooting

- For geospatial steps, verify CRS consistency between inputs and any configured projection in [`scripts/utils/projection.py`](../scripts/utils/projection.py).
- Large LiDAR/imagery may require increased memory; consider processing in an appropriate size of tiles/batches as a tiny input block may also result in an inaccurate ground point filter.
- Inspect intermediate results to make sure preset parameters fit your customized dataset.