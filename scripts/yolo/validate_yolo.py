import json
import os
import sys
from argparse import ArgumentParser
from loguru import logger
from time import time
from yaml import FullLoader, load

from pandas import concat
from ultralytics import YOLO

sys.path.insert(1, 'scripts')
from utils.constants import YOLO_TRAINING_PARAMS
from utils.misc import fill_path, format_logger

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

WORKING_DIR, MODEL, PROJECT = fill_path([WORKING_DIR, MODEL, PROJECT])

os.chdir(WORKING_DIR)
os.makedirs(os.path.join(PROJECT, 'val'), exist_ok=True)
os.makedirs(os.path.join(PROJECT, 'tst'), exist_ok=True)
filepath=os.path.join(PROJECT, 'metrics.csv')

model = YOLO(MODEL)

logger.info(f"Perform validation...")
metrics = model.val(plots=True, project=PROJECT, name='val', exist_ok=True, **YOLO_TRAINING_PARAMS)
metrics_df = metrics.to_df()

logger.info(f"Perform test...")
metrics = model.val(split='test', plots=True, project=PROJECT, name='tst', exist_ok=True, **YOLO_TRAINING_PARAMS)
metrics_df = concat([metrics_df, metrics.to_df()], ignore_index=True)

logger.info(f"Save metrics to {filepath}...")
metrics_df['dataset'] = ['val', 'tst']
metrics_df.to_csv(filepath, index=False)

logger.success(f"Done in {round(time() - tic, 2)} seconds.")