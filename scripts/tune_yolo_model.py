# https://towardsdatascience.com/the-comprehensive-guide-to-training-and-running-yolov8-models-on-custom-datasets-22946da259c3/
# https://www.digitalocean.com/community/tutorials/train-yolov5-custom-data#hyperparameter-config-file

import os
from argparse import ArgumentParser
from loguru import logger
from time import time
from yaml import load, FullLoader

from ultralytics import YOLO

from utils.constants import TILE_SIZE
from utils.misc import format_logger

import torch

logger = format_logger(logger)

def val_interval(trainer, interval=100):
    trainer.args.val = ((trainer.epoch + 1) % interval) == 0

parser = ArgumentParser(description="This script prepares COCO datasets.")
parser.add_argument('config_file', type=str, help='a YAML config file')
args = parser.parse_args()

tic = time()
logger.info('Starting...')

logger.info(f"Using {args.config_file} as config file.")

with open(args.config_file) as fp:
    cfg = load(fp, Loader=FullLoader)[os.path.basename(__file__)]

WORKING_DIR = cfg['working_directory']
# OUTPUT_DIR = cfg['output_folder']
# SAMPLE_TAGGED_IMG_SUBDIR = os.path.join(OUTPUT_DIR, cfg['sample_tagged_img_subfolder'])

YOLO_FILE = cfg['yolo_file']
MODEL = cfg['model']    # yolo11m-seg.pt
# RESUME_TRAINING = False
PARAMETERS = cfg['params']

os.chdir(WORKING_DIR)
# os.makedirs(OUTPUT_DIR, exist_ok=True)
# os.remove(SAMPLE_TAGGED_IMG_SUBDIR)
# os.makedirs(SAMPLE_TAGGED_IMG_SUBDIR)


print(f"Available GPUs: {torch.cuda.device_count()}")

model = YOLO(MODEL)

search_space = {
    "lr0": (1e-3, 1e-1),
    "lrf": (0.01, 1.0),
    "hsv_h": (0.0, 0.9),
    "hsv_s": (0.0, 0.9),
    "hsv_v": (0.0, 0.9),
}

model.tune(
    data=YOLO_FILE,
    iterations=100,
    space=search_space,
    imgsz=TILE_SIZE,
    **PARAMETERS
)
