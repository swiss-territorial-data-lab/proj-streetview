import cv2
import json
import os
import sys
from argparse import ArgumentParser
from loguru import logger
from time import time
from tqdm import tqdm
from yaml import load, FullLoader

import pandas as pd
from math import ceil

from detectron2.data.datasets import load_coco_json, register_coco_instances

from utils.misc import format_logger

logger = format_logger(logger)


def create_coco_dict(images, annotations, categories, license):
    coco_dict = {}
    coco_dict["images"] = json.loads(images.to_json(orient="records"))
    coco_dict["annotations"] = json.loads(annotations.to_json(orient="records"))
    coco_dict["categories"] = categories.copy()
    coco_dict["licenses"] = license.copy()

    return coco_dict

def remove_discarded_tiles(all_tiles_df, selectd_tiles, output_dir):
    for tile in all_tiles_df.file_name.unique():
        if tile not in selectd_tiles:
            os.remove(os.path.join(output_dir, tile))


def main(cfg_file_path):
    
    tic = time()
    logger.info('Starting...')

    logger.info(f"Using {cfg_file_path} as config file.")

    with open(cfg_file_path) as fp:
        cfg = load(fp, Loader=FullLoader)[os.path.basename(__file__)]

    WORKING_DIR = cfg['working_directory']
    OUTPUT_DIR = cfg['output_dir']
    IMAGE_DIR = cfg['image_dir']
    TILE_SIZE = cfg['tile_size']

    COCO_FILES_DICT = cfg['COCO_files']
    LICENSE = cfg['license']
    CATEGORIES = cfg['categories']
    RATIO_WO_ANNOTATIONS = cfg['ratio_wo_annotations']
    SEED = cfg['seed']
    OVERWRITE_IMAGES = cfg['overwrite_images']
    DEBUG = True

    os.chdir(WORKING_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    OUTPUT_DIR_IMAGES = "images"
    os.makedirs(os.path.join(OUTPUT_DIR, OUTPUT_DIR_IMAGES), exist_ok=True)

    # Read full COCO dataset
    images_and_annotations_df = pd.DataFrame()
    annotations_df = pd.DataFrame()
    for dataset_key, coco_file in COCO_FILES_DICT.items():
        register_coco_instances(dataset_key, {}, coco_file, "")

        coco_data = load_coco_json(coco_file, IMAGE_DIR, dataset_key)
        images_and_annotations_df = pd.concat((images_and_annotations_df, pd.DataFrame(coco_data)), ignore_index=True)

    if DEBUG:
        logger.info("Debug mode activated. Only first 100 images are processed.")
        images_and_annotations_df = images_and_annotations_df.head(100)

    # Iterate through images and clip them into tiles
    overlap_x = 96   # ok values: 96 px (19 tiles) or 200 px (25 tiles)
    overlap_y = 176
    padding = 736
    image_id = 0
    annotation_id = 0
    tiles_df = pd.DataFrame()
    clipped_annotations_df = pd.DataFrame()
    tot_tiles_with_ann = 0
    tot_tiles_without_ann = 0
    for image in tqdm(images_and_annotations_df.itertuples(), desc="Clipping images and annotations into tiles", total=len(images_and_annotations_df)):
        img = cv2.imread(image.file_name)
        if img is None:
            logger.error(f"Image {image.file_name} not found")
            continue
        h, w = img.shape[:2]
        tiles = []
        # Clip image
        for i in range(padding, h - padding, TILE_SIZE - overlap_y):
            for j in range(0, w - overlap_x, TILE_SIZE - overlap_x):
                new_filename = os.path.join(OUTPUT_DIR_IMAGES, f"{os.path.basename(image.file_name).rstrip('.jpg')}_{j}_{i}.jpg")
                tile = img[i:i+TILE_SIZE, j:j+TILE_SIZE]
                assert tile.shape[0] == TILE_SIZE and tile.shape[1] == TILE_SIZE, "Tile shape not 512 x 512 px"
                tiles.append({"height": TILE_SIZE, "width": TILE_SIZE, "id": image_id, "file_name": new_filename})

                if os.path.exists(os.path.join(OUTPUT_DIR, new_filename)) and not OVERWRITE_IMAGES:
                    continue
                cv2.imwrite(os.path.join(OUTPUT_DIR, new_filename), tile)
                image_id += 1

        all_tiles_df = pd.DataFrame(tiles)

        # Clip annotations to tiles
        annotations = []
        initial_annotation_nbr = annotation_id
        for ann in image.annotations:
            for tile in tiles:
                tile_min_x = int(tile["file_name"].split("_")[-2])
                tile_max_x = tile_min_x + TILE_SIZE
                tile_min_y = int(tile["file_name"].split("_")[-1].rstrip(".jpg"))
                tile_max_y = tile_min_y + TILE_SIZE

                # Check if annotation is outside tile
                ann_origin_x, ann_origin_y, ann_width, ann_height = ann["bbox"]
                if ann_origin_x >= tile_max_x or ann_origin_x + ann_width <= tile_min_x or ann_origin_y >= tile_max_y or ann_origin_y + ann_height <= tile_min_y:
                    continue
                # else, adapt coordinates and clip if necessary
                x1 = max(ann_origin_x - tile_min_x, 0)
                y1 = max(ann_origin_y - tile_min_y, 0)
                x2 = ann_width if x1 + ann_width <= TILE_SIZE else TILE_SIZE - x1
                y2 = ann_height if y1 + ann_height <= TILE_SIZE else TILE_SIZE - y1
                assert (all(value <= TILE_SIZE and value >= 0 for value in [x1, y1, x2, y2])), "Annotation outside tile"
                coords = ann["segmentation"][0].copy()
                new_coords_x = []
                new_coords_y = []
                for i in range(0, len(coords), 2):  # Could be improved by cutting the mask instead of flattening it
                    new_x = max(min(coords[i]-tile_min_x, TILE_SIZE), 0)
                    new_y = max(min(coords[i+1]-tile_min_y, TILE_SIZE), 0)
                    new_coords_x.append(new_x)
                    new_coords_y.append(new_y)
                    coords[i] = new_x
                    coords[i+1] = new_y
                assert (all(value <= TILE_SIZE and value >= 0 for value in coords)), "Mask outside tile"
                if all(value <=10 and value >= 500 for value in new_coords_x) or all(value <=10 and value >= 500 for value in new_coords_y):
                    logger.info("annotation on tile border")
                    continue
                annotations.append(dict(
                    id=annotation_id,
                    image_id=tile["id"],
                    category_id=1,  # Currently, single class
                    iscrowd=ann["iscrowd"],
                    bbox=[x1, y1, x2, y2],
                    segmentation=[coords]
                ))
                annotation_id += 1

        assert len(annotations) <= annotation_id - initial_annotation_nbr, "Some annotations were missed, the id did not increase enough."

        if RATIO_WO_ANNOTATIONS != 0 and len(annotations) == 0:
                tiles_df = pd.concat((tiles_df, all_tiles_df.sample(n=2, random_state=SEED)), ignore_index=True)
                tot_tiles_without_ann += 2

                remove_discarded_tiles(all_tiles_df, all_tiles_df.sample(n=2, random_state=SEED).file_name.unique(), OUTPUT_DIR)
            
        else: 
            tile_annotations_df = pd.DataFrame(annotations)
            clipped_annotations_df = pd.concat((clipped_annotations_df, tile_annotations_df), ignore_index=True)

            # Separate tiles w/ and w/o annotations
            tile_ids_w_annotations = tile_annotations_df["image_id"].unique()
            tiles_with_ann_df = all_tiles_df[all_tiles_df["id"].isin(tile_ids_w_annotations)]
            if RATIO_WO_ANNOTATIONS != 0:
                tiles_without_ann_df = all_tiles_df[~all_tiles_df["id"].isin(tile_ids_w_annotations)]

                tot_tiles_with_ann += tiles_with_ann_df.shape[0]
                nbr_tiles_without_ann = min(ceil(len(tiles_with_ann_df) * RATIO_WO_ANNOTATIONS/(1 - RATIO_WO_ANNOTATIONS)), len(tiles_without_ann_df))
                tot_tiles_without_ann += nbr_tiles_without_ann

                tiles_df = pd.concat((tiles_df, tiles_with_ann_df, tiles_without_ann_df.sample(n=nbr_tiles_without_ann, random_state=SEED)), ignore_index=True)

                remove_discarded_tiles(tiles_without_ann_df, tiles_without_ann_df.sample(n=nbr_tiles_without_ann, random_state=SEED).file_name.unique(), OUTPUT_DIR)

            else:
                tiles_df = pd.concat((tiles_df, tiles_with_ann_df), ignore_index=True)
                tot_tiles_with_ann += all_tiles_df.shape[0]

                remove_discarded_tiles(all_tiles_df, tiles_with_ann_df.file_name.unique(), OUTPUT_DIR)
                


    del annotations_df, all_tiles_df, tiles_with_ann_df, tiles_without_ann_df
    logger.info(f"Found {tot_tiles_with_ann} tiles with annotations and kept {tot_tiles_without_ann} tiles without annotations.")

    # Split tiles into train, val and test sets based on ratio 70% / 15% / 15%
    trn_tiles = tiles_df.sample(frac=0.7, random_state=SEED)
    val_tiles = tiles_df[~tiles_df["id"].isin(trn_tiles["id"])].sample(frac=0.5, random_state=SEED)
    tst_tiles = tiles_df[~tiles_df["id"].isin(trn_tiles["id"].to_list() + val_tiles["id"].to_list())]

    logger.info(f"Found {len(trn_tiles)} tiles in train set, {len(val_tiles)} tiles in val set and {len(tst_tiles)} tiles in test set.")

    dataset_tiles_dict = {"trn": trn_tiles, "val": val_tiles, "tst": tst_tiles}
    for dataset in dataset_tiles_dict.keys():
        # Split annotations
        dataset_annotations = clipped_annotations_df[clipped_annotations_df["image_id"].isin(dataset_tiles_dict[dataset]["id"])]
        logger.info(f"Found {len(dataset_annotations)} annotations in the {dataset} dataset.")

        # Create COCO dicts
        COCO_dict = create_coco_dict(dataset_tiles_dict[dataset], dataset_annotations, CATEGORIES, LICENSE)

        # Create COCO files
        logger.info(f"Creating COCO file for {dataset} set.")
        with open(os.path.join(OUTPUT_DIR, f"COCO_{dataset}.json"), "w") as fp:
            json.dump(COCO_dict, fp, indent=4)

        
if __name__ == "__main__":
    
    parser = ArgumentParser(description="This script prepares COCO datasets.")
    parser.add_argument('config_file', type=str, help='a YAML config file')
    args = parser.parse_args()

    main(args.config_file) 