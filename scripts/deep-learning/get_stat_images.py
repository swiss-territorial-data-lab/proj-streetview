import cv2
import os
import sys
from argparse import ArgumentParser
from glob import glob
from loguru import logger
from time import time
from tqdm import tqdm
from yaml import load, FullLoader

import numpy as np
import pandas as pd

sys.path.insert(1, 'scripts')
from utils.misc import format_logger

logger = format_logger(logger)


def main(cfg_file_path):

    tic = time()
    logger.info('Starting...')

    logger.info(f"Using {cfg_file_path} as config file.")

    with open(cfg_file_path) as fp:
        cfg = load(fp, Loader=FullLoader)[os.path.basename(__file__)]

    WORKING_DIR = cfg['working_directory']  
    INPUT_DIR = cfg['input_folder']
    
    os.chdir(WORKING_DIR)
    images_list = glob(INPUT_DIR + "*.jpg")

    image_specs_dict = {"image_name": [], "width": [], "height": [], "black_padding_height": []}
    for img_path in tqdm(images_list, desc="Get image specifications"):
        img = cv2.imread(img_path)
        
        image_specs_dict["image_name"].append(img_path)
        image_specs_dict["width"].append(img.shape[1])
        image_specs_dict["height"].append(img.shape[0])

        # Count number of all black pixels at the bottom of the image
        black_rows = np.sum(np.all(np.flipud(img) == 0, axis=(1, 2)))
        image_specs_dict["black_padding_height"].append(black_rows)

    image_specs_df = pd.DataFrame(image_specs_dict)
    print()

    numerical_cols = image_specs_df.select_dtypes(include=['int64', 'float64']).columns
    stat_images_df = image_specs_df[numerical_cols].describe()
    stat_images_df.to_csv(os.path.join(INPUT_DIR, "stat_images.csv"))
    logger.info(f"Done in {time() - tic:.2f} seconds.")


if __name__ == '__main__':

    parser = ArgumentParser("This script get the main info out of the panoramic images.")
    parser.add_argument("cfg_file_path", type=str, help="Path to the config file")
    args = parser.parse_args()

    main(args.cfg_file_path)