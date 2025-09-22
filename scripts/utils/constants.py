DONE_MSG = "...done."
SCATTER_PLOT_MODE = 'markers+lines'

CATEGORIES = [{'id': 1, 'name': 'manhole', 'supercategory': 'round plate'}]
TILE_SIZE = 512
IMAGE_DIR = {
    'RCNE': '/mnt/s3/proj-streetview/02_data/01_initial/01_images/NE_2020_pano/',
    'SZH': '/mnt/s3/proj-streetview/02_data/01_initial/01_images/Innovitas_2024/'
}

DETECTRON_FOLDER = 'detectron2/rcne'

COCO_FOR_YOLO_FOLDER = 'coco_for_yolo/both_new'
YOLO_DATASET = 'both_new_training'
YOLO_TRAINING_PARAMS = {
    'data': '/mnt/data-volume-02/gsalamin/GitHub/proj-streetview/config/yolo_dataset.yaml',
    'imgsz': TILE_SIZE,
    'batch': 25,
    'patience': 10,
    'multi_scale': True,
    'translate': 0,
    'single_cls': True,
    'overlap_mask': False
}
BEST_YOLO_PARAMS = {
    'epochs': 75,
    'lr0': 0.003,
    'model': 'yolo11m-seg',
    'optimizer': 'SGD'
}
MODEL_FOLDER = 'run06final'
