import os
import json
from argparse import ArgumentParser
from loguru import logger
from time import time
from tqdm import tqdm
from yaml import load, FullLoader

import pandas as pd
import shapely as shp
from geopandas import GeoDataFrame

import transform_detections as trans_dets
import utils.misc as misc
from utils.constants import CATEGORIES

logger = misc.format_logger(logger)

def main(cfg_file_path):

    tic = time()
    logger.info('Starting...')

    logger.info(f"Using {cfg_file_path} as config file.")

    with open(cfg_file_path) as fp:
        cfg = load(fp, Loader=FullLoader)[os.path.basename(__file__)]

    WORKING_DIR = cfg['working_directory']
    OUTPUT_DIR = cfg['output_folder']
    IMAGE_DIR = cfg['image_dir']
    CLIPPED_LABELS_FILES = cfg['labels_files']
    PANOPTIC_COCO_FILES = cfg['panoptic_coco_files']

    BUFFER = 1

    os.chdir(WORKING_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    logger.info(f"Read datasets...")
    clipped_labels_df = pd.DataFrame()
    for dataset_key, path in CLIPPED_LABELS_FILES.items():
        with open(path) as fp:
            coco_data = json.load(fp)
            tiles_df = pd.DataFrame.from_records(coco_data['images']).rename(columns={'id': 'image_id'})
            dataset_labels_df = pd.DataFrame.from_records(coco_data['annotations'])
        dataset_labels_df = dataset_labels_df.merge(tiles_df[['file_name', 'image_id']], how='left', on='image_id')
        dataset_labels_df['dataset'] = dataset_key
        clipped_labels_df = pd.concat([clipped_labels_df, dataset_labels_df], ignore_index=True)

    images_df = pd.DataFrame()
    for coco_file in PANOPTIC_COCO_FILES.values():
        with open(coco_file) as fp:
            coco_data = json.load(fp)
        images_df = pd.concat((images_df, pd.DataFrame.from_records(coco_data['images']).rename(columns={'id': 'image_id'})), ignore_index=True)

    del coco_data, dataset_labels_df, tiles_df

    transformed_labels= []
    for tile_name in tqdm(clipped_labels_df['file_name'].unique(), desc="Tranform labels back to panoptic images"):
        transformed_labels.extend(trans_dets.transform_annotations(tile_name, clipped_labels_df, images_df, images_dir=IMAGE_DIR, buffer=BUFFER))

    transformed_labels_gdf = GeoDataFrame(pd.DataFrame.from_records(transformed_labels), geometry='buffered_geometry')

    for dataset in CLIPPED_LABELS_FILES.keys():
        logger.info(f"Working on {dataset} dataset...")
        subset_transformed_labels_gdf = transformed_labels_gdf[transformed_labels_gdf.dataset==dataset].copy()

        logger.info('Groupping overlapping labels...')
        groupped_pairs_df = trans_dets.group_annotations(subset_transformed_labels_gdf)

        merged_labels = []
        for group in tqdm(groupped_pairs_df.group_id.unique(), desc="Merge labels in groups"):
            merged_labels.append(trans_dets.make_new_annotation(group, groupped_pairs_df, buffer=BUFFER))

        logger.info("Transforming labels to COCO format...")
        subset_images_df = images_df[images_df.image_id.isin(subset_transformed_labels_gdf.image_id.unique())].rename(columns={'image_id': 'id'})
        CATEGORIES[0]['id'] = 0 # COCO usually starts with 1, but detectron2 starts with 0
        coco_dict = misc.assemble_coco_json(subset_images_df, merged_labels, CATEGORIES)

        # Save to coco json
        filepath = os.path.join(OUTPUT_DIR, f'{dataset}_COCO_panoptic_labels.json')
        with open(filepath, 'w') as fp:
            json.dump(coco_dict, fp)
        logger.info(f'Detections saved to {filepath}.')

    toc = time()
    logger.info(f"Finished in {toc - tic:.2f} seconds.")

    
if __name__ == "__main__":
    
    parser = ArgumentParser(description="This script prepares COCO datasets.")
    parser.add_argument('config_file', type=str, help='a YAML config file')
    args = parser.parse_args()

    main(args.config_file) 