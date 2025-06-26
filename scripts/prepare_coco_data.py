import cv2
import json
import os
from argparse import ArgumentParser
from loguru import logger
from time import time
from tqdm import tqdm
from yaml import load, FullLoader

import pandas as pd
from math import ceil

from detectron2.data.datasets import load_coco_json, register_coco_instances

from utils.misc import format_logger, segmentation_to_polygon

logger = format_logger(logger)

def check_bbox_plausibility(new_origin, length, tile_size=512):
    if new_origin < 0:
        length = length + new_origin
        new_origin = 0
        if length > tile_size:
            length = tile_size
    elif new_origin + length > tile_size:
        length = tile_size - new_origin

    assert all(value <= tile_size and value >= 0 for value in [new_origin, length]), "Annotation outside tile"

    return new_origin, length


def compute_polygon_area(segm):
    poly = segmentation_to_polygon(segm)

    return poly.area


def get_new_coordinate(initial_coor, tile_min, tile_size=512):
    return max(min(initial_coor-tile_min, tile_size), 0)


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
    DEBUG = False

    os.chdir(WORKING_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    OUTPUT_DIR_IMAGES = "images"
    os.makedirs(os.path.join(OUTPUT_DIR, OUTPUT_DIR_IMAGES), exist_ok=True)
    written_files = []

    # Read full COCO dataset
    images_and_annotations_df = pd.DataFrame()
    for dataset_key, coco_file in COCO_FILES_DICT.items():
        register_coco_instances(dataset_key, {}, coco_file, "")

        coco_data = load_coco_json(coco_file, IMAGE_DIR, dataset_key)
        images_and_annotations_df = pd.concat((images_and_annotations_df, pd.DataFrame(coco_data)), ignore_index=True)

    if DEBUG:
        logger.info("Debug mode activated. Only first 100 images are processed.")
        images_and_annotations_df = images_and_annotations_df.head(100)

    # Iterate through images and clip them into tiles
    overlap_x = 224
    overlap_y = 224
    padding_y = 736
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

        # Clip image
        h, w = img.shape[:2]
        tiles = []
        for i in range(padding_y, h - padding_y, TILE_SIZE - overlap_y):
            for j in range(0, w - overlap_x, TILE_SIZE - overlap_x):
                new_filename = os.path.join(OUTPUT_DIR_IMAGES, f"{os.path.basename(image.file_name).rstrip('.jpg')}_{j}_{i}.jpg")
                tile = img[i:i+TILE_SIZE, j:j+TILE_SIZE]
                assert tile.shape[0] == TILE_SIZE and tile.shape[1] == TILE_SIZE, "Tile shape not 512 x 512 px"
                tiles.append({"height": TILE_SIZE, "width": TILE_SIZE, "id": image_id, "file_name": new_filename})
                image_id += 1

                if not os.path.exists(os.path.join(OUTPUT_DIR, new_filename)) or OVERWRITE_IMAGES:
                    cv2.imwrite(os.path.join(OUTPUT_DIR, new_filename), tile)

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

                # else, scale coordinates and clip if necessary
                # bbox
                x1, new_width = check_bbox_plausibility(ann_origin_x - tile_min_x, ann_width)
                y1, new_height = check_bbox_plausibility(ann_origin_y - tile_min_y, ann_height)                

                # segmentation
                old_coords = ann["segmentation"][0]
                coords = [get_new_coordinate(old_coords[0], tile_min_x), get_new_coordinate(old_coords[1], tile_min_y)] # set first coordinates
                new_coords_tuples = []
                for i in range(2, len(old_coords), 2):
                    new_x = get_new_coordinate(old_coords[i], tile_min_x)
                    new_y = get_new_coordinate(old_coords[i+1], tile_min_y)
                    if new_x in [0, TILE_SIZE] and coords[-2] == new_x or new_y in [0, TILE_SIZE] and coords[-1] == new_y or (new_x, new_y) in new_coords_tuples:
                        continue 
                    new_coords_tuples.append((new_x, new_y))
                    coords.extend([new_x, new_y])
                assert all(value <= TILE_SIZE and value >= 0 for value in coords), "Mask outside tile"
                if all(value[0] <=10 or value[0] >= 500 for value in new_coords_tuples) or all(value[1] <=10 or value[1] >= 500 for value in new_coords_tuples):
                    continue

                annotations.append(dict(
                    id=annotation_id,
                    image_id=tile["id"],
                    category_id=1,  # Currently, single class
                    iscrowd=ann["iscrowd"],
                    bbox=[x1, y1, new_width, new_height],
                    area=compute_polygon_area([coords]),
                    segmentation=[coords]
                ))
                annotation_id += 1

        assert len(annotations) <= annotation_id - initial_annotation_nbr, "Some annotations were missed, the id did not increase enough."

        if RATIO_WO_ANNOTATIONS != 0 and len(annotations) == 0:
                tiles_df = pd.concat((tiles_df, all_tiles_df.sample(n=2, random_state=SEED)), ignore_index=True)
                tot_tiles_without_ann += 2

                remove_discarded_tiles(all_tiles_df, all_tiles_df.sample(n=2, random_state=SEED).file_name.unique(), OUTPUT_DIR)
            
        else: 
            tile_annotations_df = pd.DataFrame(
                annotations,
                columns=['image_id'] if len(annotations) == 0 else annotations[0].keys()
            )
            clipped_annotations_df = pd.concat([clipped_annotations_df, tile_annotations_df], ignore_index=True)

            # Separate tiles w/ and w/o annotations
            condition_annotations = all_tiles_df["id"].isin(tile_annotations_df["image_id"].unique())
            tiles_with_ann_df = all_tiles_df[condition_annotations]
            tot_tiles_with_ann += tiles_with_ann_df.shape[0]

            if RATIO_WO_ANNOTATIONS != 0:
                nbr_tiles_without_ann = ceil(len(tiles_with_ann_df) * RATIO_WO_ANNOTATIONS/(1 - RATIO_WO_ANNOTATIONS))
            
                tiles_without_ann_df = all_tiles_df[~condition_annotations].sample(
                    n=min(nbr_tiles_without_ann, len(all_tiles_df[~condition_annotations])), random_state=SEED
                )
                tot_tiles_without_ann += tiles_without_ann_df.shape[0]

                tiles_df = pd.concat((tiles_df, tiles_with_ann_df, tiles_without_ann_df), ignore_index=True)

                remove_discarded_tiles(
                    all_tiles_df[~condition_annotations], tiles_without_ann_df.file_name.unique(), OUTPUT_DIR
                )

            else:
                tiles_df = pd.concat((tiles_df, tiles_with_ann_df), ignore_index=True)

                remove_discarded_tiles(all_tiles_df, tiles_with_ann_df.file_name.unique(), OUTPUT_DIR)

            del tile_annotations_df
                
    del all_tiles_df
    logger.info(f"Found {tot_tiles_with_ann} tiles with annotations and kept {tot_tiles_without_ann} tiles without annotations.")

    duplicates = clipped_annotations_df.drop(columns='id').astype({'bbox': str, 'segmentation': str}, copy=True).duplicated()
    if any(duplicates):
        logger.warning(f"Found {duplicates.sum()} duplicated annotations with different ids. Removing them...")
        clipped_annotations_df = clipped_annotations_df[~duplicates].reset_index(drop=True)

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
        coco_dict = {}
        coco_dict["images"] = json.loads(dataset_tiles_dict[dataset].to_json(orient="records"))
        coco_dict["annotations"] = json.loads(dataset_annotations.to_json(orient="records"))
        coco_dict["categories"] = CATEGORIES.copy()
        coco_dict["licenses"] = LICENSE.copy()

        # Create COCO files
        logger.info(f"Creating COCO file for {dataset} set.")
        with open(os.path.join(OUTPUT_DIR, f"COCO_{dataset}.json"), "w") as fp:
            json.dump(coco_dict, fp, indent=4)
        written_files.append(os.path.join(OUTPUT_DIR, f"COCO_{dataset}.json"))

    logger.success("Done! The following files have been created:")
    for file in written_files:
        logger.success(file)
    logger.success(f"In addition, some tiles were written in {os.path.join(OUTPUT_DIR, OUTPUT_DIR_IMAGES)}.")

    logger.info(f"Done in {round(time() - tic, 2)} seconds.")

        
if __name__ == "__main__":
    
    parser = ArgumentParser(description="This script prepares COCO datasets.")
    parser.add_argument('config_file', type=str, help='a YAML config file')
    args = parser.parse_args()

    main(args.config_file) 