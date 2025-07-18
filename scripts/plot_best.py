import os
import json
from argparse import ArgumentParser
from loguru import logger
from time import time
from yaml import load, FullLoader

import pandas as pd

from utils.misc import format_logger

logger = format_logger(logger)

def compute_f1(precision, recall):
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)

tic = time()
logger.info('Starting...')

parser = ArgumentParser(description="This script plots the results of the ray optimization for yolo hyperparameters.")
parser.add_argument('config_file', type=str, help='a YAML config file')
args = parser.parse_args()

logger.info(f"Using {args.config_file} as config file.")

with open(args.config_file) as fp:
    cfg = load(fp, Loader=FullLoader)[os.path.basename(__file__)]

# Define base directory
WORKING_DIR = cfg['working_directory']
RAY_RESULTS_DIR = cfg['ray_results_dir']

os.chdir(WORKING_DIR)

# List all sub / subdirectories (each corresponding to a different trial)
train_dirs = []
for d in os.listdir(RAY_RESULTS_DIR):
    if d.startswith("train_yolo_"):
        train_dirs += [
            os.path.join(RAY_RESULTS_DIR, d, dd) for dd in os.listdir(os.path.join(RAY_RESULTS_DIR, d)) 
            if dd.startswith("train_yolo_")
        ]
logger.info(f"Found {len(train_dirs)} training subdirectories.")

# Dictionaries to store best metrics per run
best_precision_results = {}
best_fscore_results = {}
best_map50_results = {}

# Iterate over each training subdirectory
for train_path in train_dirs:
    # train_path = os.path.join(ray_results_dir, train_dir)
    train_dir = os.path.basename(train_path)

    # Path for progress.csv
    progress_csv_path = os.path.join(train_path, "progress.csv")

    if os.path.exists(progress_csv_path):
        try:
            df = pd.read_csv(progress_csv_path)

            # Compute F1 scores for all epochs
            df["F1(M)"] = df.apply(lambda row: compute_f1(row["metrics/precision(M)"], row["metrics/recall(M)"]), axis=1)

            # Get best precision, F1-score, and mAP50
            best_precision = df["metrics/precision(M)"].max()
            best_fscore = df["F1(M)"].max()
            best_map50 = df["metrics/mAP50(M)"].max()

            # Store best results
            best_precision_results[train_dir] = best_precision
            best_fscore_results[train_dir] = best_fscore
            best_map50_results[train_dir] = best_map50

        except Exception as e:
            logger.info(f"Error reading {progress_csv_path}: {e}")

# Find best runs for precision, F-score, and mAP50
if best_precision_results:
    best_precision_run = max(best_precision_results, key=best_precision_results.get)
    best_precision_value = best_precision_results[best_precision_run]

    logger.info(f"Best Precision Run: {best_precision_run}")
    logger.info(f"Best Precision(M): {best_precision_value:.4f}")
else:
    logger.info("⚠ No valid precision results found.")

if best_fscore_results:
    best_fscore_run = max(best_fscore_results, key=best_fscore_results.get)
    best_fscore_value = best_fscore_results[best_fscore_run]

    logger.info(f"Best F-score Run: {best_fscore_run}")
    logger.info(f"Best F1(M): {best_fscore_value:.4f}")
else:
    logger.info("⚠ No valid F-score results found.")

if best_map50_results:
    best_map50_run = max(best_map50_results, key=best_map50_results.get)
    best_map50_value = best_map50_results[best_map50_run]

    logger.info(f"Best mAP50 Run: {best_map50_run}")
    logger.info(f"Best mAP50(M): {best_map50_value:.4f}")
else:
    logger.info("⚠ No valid mAP50 results found.")

logger.success(f"Done in {round(time() - tic, 2)} seconds.")