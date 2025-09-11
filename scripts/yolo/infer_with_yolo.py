import json
import os
import sys
from argparse import ArgumentParser
from loguru import logger
from time import time
from tqdm import tqdm
from yaml import FullLoader, load

from joblib import Parallel, delayed
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

os.chdir(WORKING_DIR)
os.makedirs(PROJECT, exist_ok=True)
written_files = []

logger.info(f'Working in folder "{WORKING_DIR}" with the model "{MODEL}"...')

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
        project=PROJECT, exist_ok=True, verbose=False, stream=True
    )

    if isinstance(images_infos_df, DataFrame):
        image_info_as_df = True
        _image_info_df = images_infos_df.copy()
        _image_info_df['file_name'] = _image_info_df['file_name'].apply(lambda x: os.path.basename(x))
    else:
        image_info_as_df = False
        image_id = None
        image_file = None

    # TODO: resolve mysterious backend error to force to use "threading" or use another parallelization library
    coco_detections = Parallel(n_jobs=25, backend="threading")(
        delayed(yolo_to_coco_annotations)(result, images_infos_df, image_info_as_df=image_info_as_df) 
        for result in tqdm(results, desc="Converting annotations")
    )
    flat_coco_detections = [item for sublist in coco_detections for item in sublist]
    logger.success(f"Done! {len(flat_coco_detections)} annotations were produced.")
    for i in tqdm(range(len(flat_coco_detections)), desc="Assigning IDs"):
        flat_coco_detections[i]['det_id'] = i

    logger.info(f"Save annotations...")
    filepath = os.path.join(PROJECT, f'YOLO_{dataset}_detections.json')
    with open(filepath, 'w') as fp:
        json.dump(flat_coco_detections, fp)

    written_files.append(filepath)

logger.success(f"{DONE_MSG} The following files were written:")
for filepath in written_files:
    logger.success(filepath)
logger.success(f"Done in {round(time() - tic, 2)} seconds.")