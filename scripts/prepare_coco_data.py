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

from utils.constants import CATEGORIES, TILE_SIZE
from utils.misc import assemble_coco_json, format_logger, segmentation_to_polygon

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


def remove_discarded_tiles(all_tiles_df, selected_tiles, directories_list):
    for output_dir in directories_list:
        for tile in all_tiles_df.file_name.unique():
            if tile not in selected_tiles:
                os.remove(os.path.join(output_dir, os.path.basename(tile)))

def write_image(image, image_name, dir_name, overwrite):
    filepath = os.path.join(dir_name, image_name)
    if not os.path.exists(filepath) or overwrite:
        cv2.imwrite(filepath, image)

def main(cfg_file_path):
    
    tic = time()
    logger.info('Starting...')

    logger.info(f"Using {cfg_file_path} as config file.")

    with open(cfg_file_path) as fp:
        cfg = load(fp, Loader=FullLoader)[os.path.basename(__file__)]

    WORKING_DIR = cfg['working_directory']
    IMAGE_DIR = cfg['image_dir']

    COCO_FILES_DICT = cfg['COCO_files']
    RATIO_WO_ANNOTATIONS = cfg['ratio_wo_annotations']
    SEED = cfg['seed']
    PREPARE_COCO = cfg['prepare_coco'] if 'prepare_coco' in cfg.keys() else False
    PREPARE_YOLO = cfg['prepare_yolo'] if 'prepare_yolo' in cfg.keys() else False
    OVERWRITE_IMAGES = cfg['overwrite_images']

    OVERLAP_X = 224
    OVERLAP_Y = 224
    PADDING_Y = 736
    DEBUG = False

    if not PREPARE_COCO and not PREPARE_YOLO:
        logger.critical("At least one of PREPARE_COCO or PREPARE_YOLO must be True.")
        sys.exit(1)

    os.chdir(WORKING_DIR)
    written_files = []

    OUTPUT_DIRS = []
    OUTPUT_DIR_IMAGES = "images"
    if PREPARE_COCO:
        COCO_DIR = cfg['coco_dir']
        os.makedirs(COCO_DIR, exist_ok=True)
        # Subfolder for COCO images
        os.makedirs(os.path.join(COCO_DIR, OUTPUT_DIR_IMAGES), exist_ok=True)
        OUTPUT_DIRS.append(os.path.join(COCO_DIR, OUTPUT_DIR_IMAGES))
    if PREPARE_YOLO:
        # YOLO conversion requires tiles to be saved in the same folder as COCO files
        YOLO_DIR = cfg['yolo_dir']
        os.makedirs(YOLO_DIR, exist_ok=True)
        OUTPUT_DIRS.append(os.path.join(YOLO_DIR))

    # Read full COCO dataset
    images_and_annotations_df = pd.DataFrame()
    for dataset_key, coco_file in COCO_FILES_DICT.items():
        with open(coco_file, 'r') as fp:
            coco_data = json.load(fp)
            
        images_df = pd.DataFrame.from_records(coco_data['images'])
        if 'image_id' not in images_df.columns:
            images_df.rename(columns={'id': 'image_id'}, inplace=True)
        images_df["annotations"] = [[] for _ in range(len(images_df))]
        for annotation in coco_data['annotations']:
            images_df.loc[images_df['image_id'] == annotation['image_id'], 'annotations'].iloc[0].append(annotation)

        images_and_annotations_df = pd.concat((images_and_annotations_df, pd.DataFrame(images_df)), ignore_index=True)

    if DEBUG:
        logger.info("Debug mode activated. Only first 100 images are processed.")
        images_and_annotations_df = images_and_annotations_df.head(100)

    logger.info(f"Found {len(images_and_annotations_df)} images.")

    logger.info("Split tiles into train, val and test sets based on ratio 70% / 15% / 15%...")
    trn_tiles = images_and_annotations_df.sample(frac=0.7, random_state=SEED)
    val_tiles = images_and_annotations_df[~images_and_annotations_df["image_id"].isin(trn_tiles["image_id"])].sample(frac=0.5, random_state=SEED)
    tst_tiles = images_and_annotations_df[~images_and_annotations_df["image_id"].isin(trn_tiles["image_id"].to_list() + val_tiles["image_id"].to_list())]

    images_and_annotations_df["dataset"] = None
    for dataset, df in {"trn": trn_tiles, "val": val_tiles, "tst": tst_tiles}.items():
        images_and_annotations_df.loc[images_and_annotations_df["image_id"].isin(df["image_id"]), "dataset"] = dataset
    assert all(images_and_annotations_df["dataset"].notna()), "Not all images were assigned to a dataset"

    logger.info(f"Found {len(trn_tiles)} tiles in train set, {len(val_tiles)} tiles in val set and {len(tst_tiles)} tiles in test set.")
    del trn_tiles, val_tiles, tst_tiles

    # Iterate through images and clip them into tiles
    image_id = 0
    annotation_id = 0
    tiles_df = pd.DataFrame()
    clipped_annotations_df = pd.DataFrame()
    tot_tiles_with_ann = 0
    tot_tiles_without_ann = 0
    for image in tqdm(images_and_annotations_df.itertuples(), desc="Clipping images and annotations into tiles", total=len(images_and_annotations_df)):

        img = cv2.imread(os.path.join(IMAGE_DIR, image.file_name))
        if img is None:
            logger.error(f"Image {image.file_name} not found")
            continue

        # Clip image
        h, w = img.shape[:2]
        tiles = []
        for i in range(PADDING_Y, h - PADDING_Y, TILE_SIZE - OVERLAP_Y):
            for j in range(0, w - OVERLAP_X, TILE_SIZE - OVERLAP_X):
                new_filename = os.path.join(OUTPUT_DIR_IMAGES, f"{os.path.basename(image.file_name).rstrip('.jpg')}_{j}_{i}.jpg")
                tile = img[i:i+TILE_SIZE, j:j+TILE_SIZE]
                assert tile.shape[0] == TILE_SIZE and tile.shape[1] == TILE_SIZE, "Tile shape not 512 x 512 px"
                tiles.append({"height": TILE_SIZE, "width": TILE_SIZE, "id": image_id, "file_name": new_filename, "dataset": image.dataset})
                image_id += 1

                if PREPARE_COCO and PREPARE_YOLO:
                    write_image(tile, new_filename, COCO_DIR, OVERWRITE_IMAGES)

                    dest_path = os.path.join(YOLO_DIR, os.path.basename(new_filename))
                    if not os.path.exists(dest_path):
                        os.link(new_filename, dest_path)

                elif PREPARE_COCO:
                    write_image(tile, new_filename, COCO_DIR, OVERWRITE_IMAGES)

                elif PREPARE_YOLO:
                    write_image(tile, os.path.basename(new_filename), YOLO_DIR, OVERWRITE_IMAGES)

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
                    id=int(annotation_id),
                    image_id=tile["id"],
                    category_id=int(1),  # Currently, single class
                    iscrowd=int(ann["iscrowd"]),
                    bbox=[x1, y1, new_width, new_height],
                    area=compute_polygon_area([coords]),
                    segmentation=[coords]
                ))
                annotation_id += 1

        assert len(annotations) <= annotation_id - initial_annotation_nbr, "Some annotations were missed, the id did not increase enough."

        if RATIO_WO_ANNOTATIONS != 0 and len(annotations) == 0:
                tiles_df = pd.concat((tiles_df, all_tiles_df.sample(n=2, random_state=SEED)), ignore_index=True)
                tot_tiles_without_ann += 2

                remove_discarded_tiles(all_tiles_df, all_tiles_df.sample(n=2, random_state=SEED).file_name.unique(), OUTPUT_DIRS)
            
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
                    all_tiles_df[~condition_annotations], tiles_without_ann_df.file_name.unique(), OUTPUT_DIRS
                )

            else:
                tiles_df = pd.concat((tiles_df, tiles_with_ann_df), ignore_index=True)

                remove_discarded_tiles(all_tiles_df, tiles_with_ann_df.file_name.unique(), OUTPUT_DIRS)

            del tile_annotations_df
                
    del all_tiles_df
    logger.info(f"Found {tot_tiles_with_ann} tiles with annotations and kept {tot_tiles_without_ann} tiles without annotations.")

    duplicates = clipped_annotations_df.drop(columns='id').astype({'bbox': str, 'segmentation': str}, copy=True).duplicated()
    if any(duplicates):
        logger.warning(f"Found {duplicates.sum()} duplicated annotations with different ids. Removing them...")
        clipped_annotations_df = clipped_annotations_df[~duplicates].reset_index(drop=True)

    dataset_tiles_dict = {
        key: tiles_df[tiles_df["dataset"] == key].drop(columns="dataset").reset_index(drop=True) 
        for key in tiles_df["dataset"].unique()
    }
    for dataset in dataset_tiles_dict.keys():
        # Split annotations
        dataset_annotations = clipped_annotations_df[clipped_annotations_df["image_id"].isin(dataset_tiles_dict[dataset]["id"])].copy()
        dataset_annotations = dataset_annotations.astype({"id": int, "category_id": int, "iscrowd": int}, copy=False)
        logger.info(f"Found {len(dataset_annotations)} annotations in the {dataset} dataset.")

        # Create COCO dicts
        coco_dict = assemble_coco_json(dataset_tiles_dict[dataset], dataset_annotations, CATEGORIES)

        if PREPARE_COCO:
            logger.info(f"Creating COCO file for {dataset} set.")
            with open(os.path.join(COCO_DIR, f"COCO_{dataset}.json"), "w") as fp:
                json.dump(coco_dict, fp, indent=4)
            written_files.append(os.path.join(COCO_DIR, f"COCO_{dataset}.json"))

        if PREPARE_YOLO:
            logger.info(f"Creating COCO file for the annotation transformation to YOLO.")
            dataset_tiles_dict[dataset]["file_name"] = [os.path.basename(f) for f in dataset_tiles_dict[dataset]["file_name"]]
            coco_dict = assemble_coco_json(dataset_tiles_dict[dataset], dataset_annotations, CATEGORIES)

            with open(os.path.join(YOLO_DIR, dataset + '.json'), 'w') as fp:
                json.dump(coco_dict, fp, indent=4)
            written_files.append(os.path.join(YOLO_DIR, dataset + '.json'))

    logger.success("Done! The following files have been created:")
    for file in written_files:
        logger.success(file)
    logger.success(f"In addition, some tiles were written in {OUTPUT_DIRS}.")

    logger.info(f"Done in {round(time() - tic, 2)} seconds.")

        
if __name__ == "__main__":
    
    parser = ArgumentParser(description="This script prepares COCO datasets.")
    parser.add_argument('config_file', type=str, help='a YAML config file')
    args = parser.parse_args()

    main(args.config_file) 