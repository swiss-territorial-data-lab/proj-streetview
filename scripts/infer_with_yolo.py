import json
import os
from argparse import ArgumentParser
from loguru import logger
from time import time
from yaml import FullLoader, load

from pandas import DataFrame
from ultralytics import YOLO

from utils.constants import TILE_SIZE
from utils.misc import format_logger
from utils.yolo_to_coco import yolo_to_coco_annotations

logger = format_logger(logger)

tic = time()
logger.info('Starting...')

parser = ArgumentParser(description="This script make inference with a yolo model.")
parser.add_argument('config_file', type=str, help='a YAML config file')
args = parser.parse_args()

logger.info(f"Using {args.config_file} as config file.")

with open(args.config_file) as fp:
    cfg = load(fp, Loader=FullLoader)[os.path.basename(__file__)]

WORKING_DIR = cfg['working_directory']
DATASET_IMAGES_DIR = cfg['dataset_images_folder']

MODEL = cfg['model']
PROJECT = cfg['project']
PROJECT_NAME = [path_part for path_part in MODEL.split('/') if 'run' in path_part][0]

COCO_INFO_DIR = cfg['coco_info_folder']

os.chdir(WORKING_DIR)
os.makedirs(os.path.join(PROJECT, PROJECT_NAME), exist_ok=True)
written_files = []

for dataset, path in DATASET_IMAGES_DIR.items():
    logger.info(f"Working on the dataset {dataset}...")
    logger.info('Get image infos...')
    with open(os.path.join(COCO_INFO_DIR, f'{dataset}.json'), 'r') as fp:
        image_infos_dict = json.load(fp)['images']
    images_infos_df = DataFrame.from_records(image_infos_dict)[['file_name', 'id']]

    logger.info(f"Perform inference...")
    model = YOLO(MODEL)
    results = model(
        path, 
        conf=0.05,
        imgsz=TILE_SIZE, retina_masks=True, 
        project=PROJECT, name=PROJECT_NAME, exist_ok=True, verbose=False, stream=True
    )

    coco_detections = yolo_to_coco_annotations(results, images_infos_df)

    logger.info(f"Save annotations...")
    filepath = os.path.join(PROJECT, PROJECT_NAME, f'YOLO_{dataset}_detections.json')
    with open(filepath, 'w') as fp:
        json.dump(coco_detections, fp)

    written_files.append(filepath)

logger.success(f"Done in {round(time() - tic, 2)} seconds.")