import os
from argparse import ArgumentParser
from loguru import logger
from time import time
from tqdm import tqdm
from yaml import load, FullLoader

import requests
from geopandas import read_file
from pandas import read_table

from utils.misc import fill_path, format_logger

logger = format_logger(logger)

tic = time()
logger.info('Starting...')

parser = ArgumentParser(description="This script gets images for the RCNE dataset.")
parser.add_argument('config_file', type=str, help='a YAML config file')
args = parser.parse_args()

logger.info(f"Using {args.config_file} as config file.")

with open(args.config_file) as fp:
    cfg = load(fp, Loader=FullLoader)[os.path.basename(__file__)]

WORKING_DIR = cfg['working_directory']
OUTPUT_DIR = cfg['output_folder']
IMAGE_INFOS = cfg['image_infos']
OVERWRITE = cfg['overwrite'] if 'overwrite' in cfg.keys() else False

WORKING_DIR, OUTPUT_DIR = fill_path([WORKING_DIR, OUTPUT_DIR])

os.chdir(WORKING_DIR)
logger.info(f'Working directory set to {WORKING_DIR}.')
os.makedirs(OUTPUT_DIR, exist_ok=True)
written_files = []

logger.info("Reading data...")
if any(IMAGE_INFOS.endswith(ext) for ext in ['xlsx', 'csv', 'txt']):
    df = read_table(IMAGE_INFOS)
elif any(IMAGE_INFOS.endswith(ext) for ext in ['json', 'shp', 'gpkg']):
    df = read_file(IMAGE_INFOS)
else:
    raise ValueError(f"Unknown file type for {IMAGE_INFOS}.")

logger.info("Downloading images...")
for image_row in tqdm(df.head(25).itertuples(), total=len(df), desc="Downloading images"):
    image_link = image_row.URL
    dest_path = os.path.join(OUTPUT_DIR, os.path.basename(image_link))
    if not os.path.exists(dest_path) or OVERWRITE:
        # Get image online and save it locally
        answer = requests.get(image_link)
        if answer.status_code == 200:
            with open(dest_path, 'wb') as fp:
                fp.write(answer.content)
            written_files.append(dest_path)
        else:
            logger.error(f"Failed to download {os.path.basename(image_link)}.")

logger.success(f"Done! {len(written_files)} images were downloaded in {os.path.join(WORKING_DIR, OUTPUT_DIR)}.")
logger.info(f"Done in {round(time() - tic, 2)} seconds.")