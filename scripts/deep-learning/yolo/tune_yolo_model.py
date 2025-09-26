# https://towardsdatascience.com/the-comprehensive-guide-to-training-and-running-yolov8-models-on-custom-datasets-22946da259c3/
# https://www.digitalocean.com/community/tutorials/train-yolov5-custom-data#hyperparameter-config-file

import os
import sys
from argparse import ArgumentParser
from loguru import logger
from time import time
from yaml import load, FullLoader

from ultralytics import YOLO

sys.path.insert(1, 'scripts')
from utils.constants import YOLO_TRAINING_PARAMS
from utils.misc import fill_path, format_logger

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
PROJECT = cfg['project']

YOLO_FILE = cfg['yolo_file']
MODEL = cfg['model']
PARAMETERS = cfg['params']

WORKING_DIR, PROJECT = fill_path([WORKING_DIR, PROJECT])
os.chdir(WORKING_DIR)

print(f"Available GPUs: {torch.cuda.device_count()}")

model = YOLO(MODEL)

search_space = {
    "lr0": (1e-3, 1e-1),
    "lrf": (0.01, 1.0),
    "hsv_h": (0.0, 0.9),
    "hsv_s": (0.0, 0.9),
    "hsv_v": (0.0, 0.9),
}

if not os.path.exists(YOLO_TRAINING_PARAMS['data']):
    YOLO_TRAINING_PARAMS['data'] = os.path.join('/app', 'config', YOLO_TRAINING_PARAMS['data'].split('config/')[-1])

model.tune(
    iterations=50,
    space=search_space,
    project=PROJECT,
    **PARAMETERS,
    **YOLO_TRAINING_PARAMS
)
