import json
import os
import sys
from argparse import ArgumentParser
from loguru import logger
from time import time
from yaml import FullLoader, load

from pandas import DataFrame
from ultralytics import YOLO

sys.path.insert(1, 'scripts')
from utils.constants import DONE_MSG, TILE_SIZE 
from utils.misc import fill_path, format_logger
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

COCO_INFO_DIR = cfg['coco_info_folder']

WORKING_DIR, COCO_INFO_DIR, MODEL, PROJECT= fill_path([WORKING_DIR, COCO_INFO_DIR, MODEL, PROJECT])
PROJECT_NAME = [path_part for path_part in MODEL.split('/') if 'run' in path_part][0]

os.chdir(WORKING_DIR)
os.makedirs(os.path.join(PROJECT, PROJECT_NAME), exist_ok=True)
written_files = []

last_id = 0
for dataset, path in DATASET_IMAGES_DIR.items():
    logger.info(f"Working on the dataset {dataset}...")
    logger.info('Get image infos...')
    with open(os.path.join(COCO_INFO_DIR, f'{dataset}.json'), 'r') as fp:
        image_infos_dict = json.load(fp)['images']
    images_infos_df = DataFrame.from_records(image_infos_dict)[['file_name', 'id']]

    logger.info(f"Perform inference...")
    model = YOLO(MODEL)
    results = model(
        fill_path(path),
        conf=0.05,
        imgsz=TILE_SIZE, retina_masks=True, 
        project=PROJECT, name=PROJECT_NAME, exist_ok=True, verbose=False, stream=True
    )

    coco_detections = yolo_to_coco_annotations(results, images_infos_df, start_id=last_id)
    last_id = coco_detections[-1]['det_id']
    logger.success(f"Done! {len(coco_detections)} annotations were produced.")

    logger.info(f"Save annotations...")
    filepath = os.path.join(PROJECT, PROJECT_NAME, f'YOLO_{dataset}_detections.json')
    with open(filepath, 'w') as fp:
        json.dump(coco_detections, fp)

    written_files.append(filepath)

logger.success(f"{DONE_MSG} The following files were written:")
for filepath in written_files:
    logger.success(filepath)
logger.success(f"Done in {round(time() - tic, 2)} seconds.")