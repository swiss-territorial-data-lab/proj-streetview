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

    COCO_FILES_DICT = cfg['COCO_files']
    RATIO_WO_ANNOTATIONS = cfg['ratio_wo_annotations']
    SEED = cfg['seed']
    PREPARE_YOLO = cfg['preapre_yolo']
    OVERWRITE_IMAGES = cfg['overwrite_images']

    OVERLAP_X = 224
    OVERLAP_Y = 224
    PADDING_Y = 736
    DEBUG = False

    os.chdir(WORKING_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    OUTPUT_DIR_IMAGES = "images"
    os.makedirs(os.path.join(OUTPUT_DIR, OUTPUT_DIR_IMAGES), exist_ok=True)
    written_files = []

    if PREPARE_YOLO:
        YOLO_DIR = 'COCO_datasets'
        os.makedirs(YOLO_DIR, exist_ok=True)

    # Read full COCO dataset
    images_and_annotations_df = pd.DataFrame()
    for dataset_key, coco_file in COCO_FILES_DICT.items():
        register_coco_instances(dataset_key, {}, coco_file, "")

        coco_data = load_coco_json(coco_file, IMAGE_DIR, dataset_key)
        images_and_annotations_df = pd.concat((images_and_annotations_df, pd.DataFrame(coco_data)), ignore_index=True)

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

        img = cv2.imread(image.file_name)
        if img is None:
            logger.error(f"Image {image.file_name} not found")
            continue

        # Clip image
        h, w = img.shape[:2]
        tiles = []
        for i in range(PADDING_Y, h - PADDING_Y, TILE_SIZE - OVERLAP_Y):
            for j in range(0, w - OVERLAP_X, TILE_SIZE - OVERLAP_X):
                new_filename = os.path.join(OUTPUT_DIR_IMAGES, f"{os.path.basename(image.file_name).rstrip('.jpg')}_{j}_{i}.jpg")
                new_filepath = os.path.join(OUTPUT_DIR, new_filename)
                tile = img[i:i+TILE_SIZE, j:j+TILE_SIZE]
                assert tile.shape[0] == TILE_SIZE and tile.shape[1] == TILE_SIZE, "Tile shape not 512 x 512 px"
                tiles.append({"height": TILE_SIZE, "width": TILE_SIZE, "id": image_id, "file_name": new_filename, "dataset": image.dataset})
                image_id += 1

                if not os.path.exists(new_filepath) or OVERWRITE_IMAGES:
                    cv2.imwrite(new_filepath, tile)
                if PREPARE_YOLO:
                    dest_path = os.path.join(YOLO_DIR, os.path.basename(new_filepath))
                    if os.path.exists(dest_path):
                        os.remove(dest_path)
                    os.link(new_filepath, dest_path)

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

        # Create COCO files
        logger.info(f"Creating COCO file for {dataset} set.")
        with open(os.path.join(OUTPUT_DIR, f"COCO_{dataset}.json"), "w") as fp:
            json.dump(coco_dict, fp, indent=4)
        written_files.append(os.path.join(OUTPUT_DIR, f"COCO_{dataset}.json"))

        if PREPARE_YOLO:
            logger.info(f"Creating corresponding COCO file for the annotation transformation to YOLO.")
            dataset_tiles_dict[dataset]["file_name"] = [os.path.basename(f) for f in dataset_tiles_dict[dataset]["file_name"]]
            coco_dict = assemble_coco_json(dataset_tiles_dict[dataset], dataset_annotations, CATEGORIES)

            with open(os.path.join(YOLO_DIR, dataset + '.json'), 'w') as fp:
                json.dump(coco_dict, fp, indent=4)
            written_files.append(os.path.join(YOLO_DIR, dataset + '.json'))

    logger.success("Done! The following files have been created:")
    for file in written_files:
        logger.success(file)
    logger.success(f"In addition, some tiles were written in {OUTPUT_DIR}.")

    logger.info(f"Done in {round(time() - tic, 2)} seconds.")

        
if __name__ == "__main__":
    
    parser = ArgumentParser(description="This script prepares COCO datasets.")
    parser.add_argument('config_file', type=str, help='a YAML config file')
    args = parser.parse_args()

    main(args.config_file) 