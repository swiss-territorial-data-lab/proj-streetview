import os
from argparse import ArgumentParser
from loguru import logger
from time import time
from yaml import load, FullLoader

from ultralytics.data.converter import convert_coco

from misc import format_logger

logger = format_logger(logger)

tic = time()
logger.info('Starting...')

parser = ArgumentParser(description="This script prepares YOLO datasets from the COCO ones.")
parser.add_argument('config_file', type=str, help='a YAML config file')
args = parser.parse_args()

logger.info(f"Using {args.config_file} as config file.")

with open(args.config_file) as fp:
    cfg = load(fp, Loader=FullLoader)[os.path.basename(__file__)]

WORKING_DIR = cfg['working_directory']
INPUT_DIR = cfg['input_folder']
OUTPUT_DIR = cfg['output_folder']

os.chdir(WORKING_DIR)
if os.path.exists(OUTPUT_DIR):
    os.system(f"rm -r {OUTPUT_DIR}")

# cf. https://github.com/ultralytics/ultralytics/blob/main/ultralytics/data/converter.py#L228
convert_coco(labels_dir=INPUT_DIR, save_dir=OUTPUT_DIR, use_segments=True)

logger.success(f"Done! YOLO dataset was created in {os.path.join(WORKING_DIR, OUTPUT_DIR)}.")
logger.info(f"Done in {round(time() - tic, 2)} seconds.")