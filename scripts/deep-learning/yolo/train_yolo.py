import json
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
PROJECT_NAME = cfg['name']
BEST_PARAMETERS_PATH = cfg['best_parameters_path']

RESUME_TRAINING = cfg['resume_training']

WORKING_DIR, BEST_PARAMETERS_PATH, PROJECT, PROJECT_NAME = fill_path([WORKING_DIR, BEST_PARAMETERS_PATH, PROJECT, PROJECT_NAME])

os.chdir(WORKING_DIR)

print(f"Available GPUs: {torch.cuda.device_count()}")

with open(BEST_PARAMETERS_PATH) as fp:
    best_parameters = json.load(fp)

model = YOLO(best_parameters['model'])
best_parameters.pop('model')

if 'batch' in best_parameters.keys():
    YOLO_TRAINING_PARAMS.pop('batch')
if 'patience' in best_parameters.keys():
    YOLO_TRAINING_PARAMS.pop('patience')

model.train(
    project=PROJECT,
    name=PROJECT_NAME,
    save=True,
    save_period=5,
    plots=True,
    resume=RESUME_TRAINING,
    **YOLO_TRAINING_PARAMS,
    **best_parameters
    )