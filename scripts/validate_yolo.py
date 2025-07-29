import json
import os
from argparse import ArgumentParser
from loguru import logger
from time import time
from yaml import FullLoader, load

from pandas import concat
from ultralytics import YOLO

from utils.constants import YOLO_TRAINING_PARAMS
from utils.misc import format_logger

logger = format_logger(logger)

tic = time()
logger.info('Starting...')

parser = ArgumentParser(description="This script outputs metrics for the validation set of a yolo model.")
parser.add_argument('config_file', type=str, help='a YAML config file')
args = parser.parse_args()

logger.info(f"Using {args.config_file} as config file.")

with open(args.config_file) as fp:
    cfg = load(fp, Loader=FullLoader)[os.path.basename(__file__)]

WORKING_DIR = cfg['working_directory']

MODEL = cfg['model']
PROJECT = cfg['project']
PROJECT_NAME = [path_part for path_part in MODEL.split('/') if 'run' in path_part][0]
BEST_PARAMETERS_PATH = cfg['best_parameters_path']

os.chdir(WORKING_DIR)
os.makedirs(os.path.join(PROJECT, PROJECT_NAME, 'val'), exist_ok=True)
os.makedirs(os.path.join(PROJECT, PROJECT_NAME, 'tst'), exist_ok=True)
filepath=os.path.join(PROJECT, PROJECT_NAME, 'metrics.csv')

model = YOLO(MODEL)
with open(BEST_PARAMETERS_PATH) as fp:
    best_parameters = json.load(fp)
    batch_size = best_parameters['batch']

logger.info(f"Perform validation...")
metrics = model.val(batch=batch_size, plots=True, project=PROJECT, name=PROJECT_NAME + '/val', exist_ok=True, **YOLO_TRAINING_PARAMS)
metrics_df = metrics.to_df()

logger.info(f"Perform test...")
metrics = model.val(batch=batch_size, split='test', plots=True, project=PROJECT, name=PROJECT_NAME + '/tst', exist_ok=True, **YOLO_TRAINING_PARAMS)
metrics_df = concat([metrics_df, metrics.to_df()], ignore_index=True)

logger.info(f"Save metrics to {filepath}...")
metrics_df.to_csv(filepath)

logger.success(f"Done in {round(time() - tic, 2)} seconds.")