import json
import os  
from argparse import ArgumentParser
from loguru import logger
from time import time
from tqdm import tqdm
from yaml import load, FullLoader

import pandas as pd

from misc import format_logger

logger = format_logger(logger)

tic = time()
logger.info('Starting...')

parser = ArgumentParser(description="This script redistributes images between datasets.")
parser.add_argument('config_file', type=str, help='a YAML config file')
args = parser.parse_args()

logger.info(f"Using {args.config_file} as config file.")

cfg_file_path = args.config_file

with open(cfg_file_path) as fp:
    cfg = load(fp, Loader=FullLoader)[os.path.basename(__file__)]

WORKING_DIR = cfg['working_directory']
IMAGE_DIR = cfg['image_folder']
OUTPUT_DIR = cfg['output_folder']
COCO_FILES_DICT = cfg['COCO_files']
OVERWRITE = cfg['overwrite']

os.chdir(WORKING_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

for dataset, coco_file in COCO_FILES_DICT.items():
    dataset_dir = os.path.join(OUTPUT_DIR, dataset)
    os.makedirs(dataset_dir, exist_ok=True)
    with open(coco_file) as fp:
        images_df = pd.DataFrame(json.load(fp)['images']).drop_duplicates(subset=["file_name"])

    for index, row in tqdm(images_df.iterrows(), desc=f"Linking images for {dataset} dataset", total=len(images_df)):
        image_path = os.path.join(IMAGE_DIR, os.path.basename(row['file_name']))
        dest_path = os.path.join(dataset_dir, os.path.basename(row['file_name']))
        if os.path.exists(dest_path):
            if not OVERWRITE:
                continue
            os.remove(dest_path)
        os.link(image_path, dest_path)

logger.success(f"Done! {len(COCO_FILES_DICT)} datasets were created in {os.path.join(WORKING_DIR, OUTPUT_DIR)}.")
logger.info(f"Done in {round(time() - tic, 2)} seconds.")