import cv2
import json
import os
import sys
from argparse import ArgumentParser
from joblib import Parallel, delayed
from loguru import logger
from time import time
from tqdm import tqdm
from yaml import load, FullLoader

import numpy as np
import pandas as pd
from itertools import product
from math import ceil

import utils.misc as misc
from utils.constants import CATEGORIES, IMAGE_DIR, TILE_SIZE

logger = misc.format_logger(logger)


def borderline_intersection(coord_tuples_list):
    first_coords = all(value[0] <= TILE_SIZE * 0.02 or value[0] >= TILE_SIZE * 0.98 for value in coord_tuples_list)
    second_coords = all(value[1] <= TILE_SIZE * 0.02 or value[1] >= TILE_SIZE * 0.98 for value in coord_tuples_list)

    return first_coords or second_coords


def check_bbox_plausibility(new_origin, length):
    """
    Adjusts and checks the plausibility of a bounding box's origin and length within a tile.

    Args:
        new_origin (int): The proposed new origin of the bounding box.
        length (int): The proposed length of the bounding box.

    Returns:
        tuple: A tuple containing the adjusted new_origin and length, ensuring they fit within the tile size.

    Raises:
        AssertionError: If the adjusted bounding box origin or length is outside the tile.
    """

    if new_origin < 0:
        length = length + new_origin
        new_origin = 0
        if length > TILE_SIZE:
            length = TILE_SIZE
    elif new_origin + length > TILE_SIZE:
        length = TILE_SIZE - new_origin

    assert all(value <= TILE_SIZE and value >= 0 for value in [new_origin, length]), "Annotation outside tile"

    return new_origin, length


def get_new_coordinate(initial_coor, tile_min):
    # Calculates the new coordinate for a bounding box annotation to be within a tile.
    return max(min(initial_coor-tile_min, TILE_SIZE), 0)


def image_to_tiles(image_path, corresponding_tiles, rejected_annotations_df, tasks_dict, overwrite=False):
    """
    Processes an image by dividing it into tiles, applying masks on pixels corresponding to rejected annotations, and saving the tiles.

    Args:
        image_path (str): The path of the image file.
        corresponding_tiles (list): A list of paths to the tiles that the image should be cut into.
        rejected_annotations_df (DataFrame): A DataFrame containing annotations that should be rejected (masked) on the tiles.
        tasks_dict (dict): A dictionary defining the tasks to prepare data for, including subfolder paths.
        output_dir (str, optional): The directory where the tiles should be saved. Defaults to 'outputs'.
        overwrite (bool, optional): Whether to overwrite existing tiles. Defaults to False.

    Returns:
        dict: A dictionary of the tiles that could not be saved with tile paths as key and False as value.
    """

    prepare_coco = tasks_dict['coco']['prepare_data'] if 'coco' in tasks_dict.keys() else False
    prepare_yolo = tasks_dict['yolo']['prepare_data'] if 'yolo' in tasks_dict.keys() else False
    output_dirs = [tasks_dict[task]['subfolder'] for task in tasks_dict.keys() if tasks_dict[task]['prepare_data']]
    achieved = {image_path: []}
    if all(os.path.exists(os.path.join(output_dir, tile_path)) and not overwrite for output_dir, tile_path in product(output_dirs,corresponding_tiles)):
        return {} 

    img = cv2.imread(os.path.join(image_path))
    if img is None:
        logger.error(f"Image {image_path} could not be read.")
        return {tile_path: False for tile_path in corresponding_tiles}

    achieved = {}
    for tile_name in corresponding_tiles:
        files_exist = all([
            os.path.exists(os.path.join(output_dir, tile_name)) or os.path.exists(os.path.join(output_dir, os.path.basename(tile_name))) 
            for output_dir in output_dirs
        ])
        if not files_exist or overwrite:
            i = int(tile_name.split("_")[-1].rstrip(".jpg"))
            j = int(tile_name.split("_")[-2])
            tile = np.zeros((TILE_SIZE, TILE_SIZE, 3), dtype=np.uint8)
            try:
                tile[:] = img[i:i+TILE_SIZE, j:j+TILE_SIZE]
            except:
                tmp = img[i:i+TILE_SIZE, j:j+TILE_SIZE]
                assert any([tmp.shape[0] == 0, tmp.shape[1]  == 0]), \
                    f"Tile {tile_name} was not prefectly cut into tiles of size {TILE_SIZE}. Left: {img.shape[1]-TILE_SIZE}, Top: {img.shape[0]-TILE_SIZE}"
                achieved[tile_name] = False
                continue

            # Draw a black mask on reject annotations
            if not rejected_annotations_df.empty:
                annotations_to_mask_df = rejected_annotations_df[rejected_annotations_df.file_name == tile_name]
                for ann in annotations_to_mask_df.itertuples():
                    bbox = [int(b) for b in ann.bbox]
                    cv2.rectangle(
                        img=tile, 
                        pt1=(bbox[0], bbox[1]), 
                        pt2=(min(bbox[0] + round(bbox[2]*1.1), TILE_SIZE), min(bbox[1] + round(bbox[3]*1.1), TILE_SIZE)), 
                        color=(0, 0, 0), 
                        thickness=-1
                    )
                
            if prepare_coco and prepare_yolo:
                tile_path = os.path.join(tasks_dict['coco']['subfolder'], tile_name)
                achieved_coco = cv2.imwrite(tile_path, tile)

                dest_path = os.path.join(tasks_dict['yolo']['subfolder'], os.path.basename(tile_name))
                if not os.path.exists(dest_path):
                    os.link(tile_path, dest_path)

                achievment = achieved_coco and os.path.exists(dest_path)

            elif prepare_coco:
                tile_path = os.path.join(tasks_dict['coco']['subfolder'], tile_name)
                achievment = cv2.imwrite(tile_path, tile)

            elif prepare_yolo:
                tile_path = os.path.join(tasks_dict['yolo']['subfolder'], os.path.basename(tile_name))
                achievment = cv2.imwrite(tile_path, tile)

            if not achievment:
                logger.error(f'Tile {tile_path} could not be produced.')
                achieved[tile_name] = False

    return achieved

def write_image(image, image_name, dir_name, overwrite):
    filepath = os.path.join(dir_name, image_name)
    if not os.path.exists(filepath) or overwrite:
        cv2.imwrite(filepath, image)

def select_low_tiles(tiles_df, clipping_params_dict, excluded_height_ratio=1/2):
    """
    Select tiles that are above a certain height ratio.

    Args:
        tiles_df (DataFrame): A DataFrame containing the tiles.
        excluded_height_ratio (float, optional): The height ratio under which tiles should be excluded. Defaults to 1/2.

    Returns:
        DataFrame: A DataFrame containing the selected tiles.
    """
    _tiles_df = tiles_df.copy()
    aoi = _tiles_df.loc[0, 'AOI']
    if "height" in clipping_params_dict[aoi].keys():
        image_height = clipping_params_dict[aoi]["height"]
    elif 'lb4' in clipping_params_dict[aoi].keys():
        image_height = clipping_params_dict[aoi]['lb4']["height"]
    else:
        image_height = clipping_params_dict[aoi]['else']["height"]

    _tiles_df["row_level"] = _tiles_df["file_name"].apply(lambda x: int(x.split("_")[-1].rstrip(".jpg")))
    low_tiles_df = _tiles_df[_tiles_df["row_level"] >= image_height*excluded_height_ratio].reset_index(drop=True)
    low_tiles_df.drop(columns=["row_level"], inplace=True)

    return low_tiles_df


def main(cfg_file_path):
    
    tic = time()
    logger.info('Starting...')

    logger.info(f"Using {cfg_file_path} as config file.")

    with open(cfg_file_path) as fp:
        cfg = load(fp, Loader=FullLoader)[os.path.basename(__file__)]

    WORKING_DIR = misc.fill_path(cfg['working_directory'])

    ORIGINAL_COCO_FILES_DICT = cfg['original_COCO_files']
    VALIDATED_COCO_FILES_DICT = cfg['validated_COCO_files'] if 'validated_COCO_files' in cfg.keys() else {}
    CLIPPING_PARAMS = cfg['clipping_params']
    RATIO_WO_ANNOTATIONS = cfg['ratio_wo_annotations']
    SEED = cfg['seed']
    OVERWRITE_IMAGES = cfg['overwrite_images']

    TASKS = cfg['tasks']
    MAKE_OTHER_DATASET = TASKS['make_other_dataset']
    TASKS.pop('make_other_dataset')
    TEST_ONLY = TASKS['test_only']
    TASKS.pop('test_only')
    PREPARE_COCO = TASKS['coco']['prepare_data'] if 'coco' in TASKS.keys() else False
    PREPARE_YOLO = TASKS['yolo']['prepare_data'] if 'yolo' in TASKS.keys() else False

    DEBUG = False

    for aoi in CLIPPING_PARAMS.keys():
        params = CLIPPING_PARAMS[aoi]
        try:
            logger.info(f"Including an overlap of {round(params['overlap_x']/TILE_SIZE*100,1)} % in the X axis and {round(params['overlap_y']/TILE_SIZE*100,1)} % in the Y axis for the {aoi} AOI.")
        except:
            for key in params.keys():
                logger.info(f"Including an overlap of {round(params[key]['overlap_x']/TILE_SIZE*100,1)} % in the X axis and {round(params[key]['overlap_y']/TILE_SIZE*100,1)} % in the Y axis for the {aoi} AOI ({key}).")

    if not PREPARE_COCO and not PREPARE_YOLO:
        logger.critical("At least one of PREPARE_COCO or PREPARE_YOLO must be True.")
        sys.exit(1)

    os.chdir(WORKING_DIR)
    written_files = []

    OUTPUT_DIRS = []
    OUTPUT_DIR_IMAGES = "images"
    if PREPARE_COCO:
        TASKS['coco']['subfolder'] = misc.fill_path(TASKS['coco']['subfolder'])
        COCO_DIR = TASKS['coco']['subfolder']
        os.makedirs(COCO_DIR, exist_ok=True)
        # Subfolder for COCO images
        os.makedirs(os.path.join(COCO_DIR, OUTPUT_DIR_IMAGES), exist_ok=True)
        OUTPUT_DIRS.append(os.path.join(COCO_DIR, OUTPUT_DIR_IMAGES))
    if PREPARE_YOLO:
        # YOLO conversion requires tiles to be saved in the same folder as COCO files
        TASKS['yolo']['subfolder'] = misc.fill_path(TASKS['yolo']['subfolder'])
        YOLO_DIR = TASKS['yolo']['subfolder']
        os.makedirs(YOLO_DIR, exist_ok=True)
        OUTPUT_DIRS.append(os.path.join(YOLO_DIR))

    logger.info(f"Read COCO files...")
    # Read full COCO dataset
    original_imgs_and_anns_dict = {}
    id_correspondence_df = pd.DataFrame()
    max_id = 0
    for aoi, coco_file in ORIGINAL_COCO_FILES_DICT.items():
        images_df = misc.read_coco_dataset(coco_file)
        images_df['original_id'] = images_df['image_id']
        # Make image IDs unique and consistent
        images_df['image_id'] = images_df['image_id'] + max_id
        original_imgs_and_anns_dict[aoi] = images_df
        max_id += images_df.image_id.max() + 1

        images_df["AOI"] = aoi
        id_correspondence_df = pd.concat([id_correspondence_df, images_df[["AOI", 'image_id', 'original_id']]], ignore_index=True)

    for OUTPUT_DIR in OUTPUT_DIRS:
        filepath = os.path.join(OUTPUT_DIR.rstrip('images') if 'images' in OUTPUT_DIR[8:] else OUTPUT_DIR, "original_ids.csv")
        id_correspondence_df.to_csv(filepath, index=False)
        written_files.append(filepath)

    # Read validated COCO dataset
    valid_imgs_and_anns_dict = {}
    if isinstance(VALIDATED_COCO_FILES_DICT, dict):
        # Case: training
        for aoi, coco_file in VALIDATED_COCO_FILES_DICT.items():
            images_df = misc.read_coco_dataset(coco_file)
            images_df['original_id'] = images_df['image_id']
            # Get unique IDs from the original COCO dataset
            images_df['image_id'] = images_df.drop(columns='image_id').merge(
                original_imgs_and_anns_dict[aoi],
                how='left', on='original_id'
            ).image_id
            assert images_df['image_id'].isna().sum() == 0, "Validated COCO dataset contains images that are not in the original COCO dataset."
            valid_imgs_and_anns_dict[aoi] = images_df
    else:
        # Case: inference-only
        valid_imgs_and_anns_dict = {key: pd.DataFrame() for key in original_imgs_and_anns_dict.keys()}

    if DEBUG:
        logger.info("Debug mode activated. Only first 100 images are processed.")
        for key in original_imgs_and_anns_dict.keys():
            original_imgs_and_anns_dict[key] = original_imgs_and_anns_dict[key].head(100)
            valid_imgs_and_anns_dict[key] = valid_imgs_and_anns_dict[key].head(200)

    logger.info(f"Found {sum([len(df) for df in valid_imgs_and_anns_dict.values()])} images for validated annotations.")

    if all(df.empty for df in valid_imgs_and_anns_dict.values()):
        logger.info("No validated annotations found. Only inference is possible.")
        MAKE_OTHER_DATASET = True
        RATIO_WO_ANNOTATIONS = 0
        for original_imgs_and_anns_df in original_imgs_and_anns_dict.values():
            original_imgs_and_anns_df['dataset'] = 'oth'
    elif TEST_ONLY:
        logger.warning('Test-only mode activated. All annotations will be stored in the test set.')
        for key, valid_imgs_and_anns_df in valid_imgs_and_anns_dict.items():
            original_imgs_and_anns_df = original_imgs_and_anns_dict[key]

            valid_imgs_and_anns_df["dataset"] = "tst"
            original_imgs_and_anns_df.loc[original_imgs_and_anns_df["image_id"].isin(valid_imgs_and_anns_df["image_id"]), "dataset"] = "tst"
            original_imgs_and_anns_df.loc[original_imgs_and_anns_df.dataset.isna(), "dataset"] = "oth"
    else:
        logger.info("Splitting images into train, val and test sets based on ratio 70% / 15% / 15%...")
        for key, valid_imgs_and_anns_df in valid_imgs_and_anns_dict.items():
            trn_tiles = valid_imgs_and_anns_df.sample(frac=0.7, random_state=SEED)
            val_tiles = valid_imgs_and_anns_df[~valid_imgs_and_anns_df["image_id"].isin(trn_tiles["image_id"])].sample(frac=0.5, random_state=SEED)
            tst_tiles = valid_imgs_and_anns_df[~valid_imgs_and_anns_df["image_id"].isin(trn_tiles["image_id"].to_list() + val_tiles["image_id"].to_list())]

            # Map dataset on the images
            original_imgs_and_anns_df = original_imgs_and_anns_dict[key]
            valid_imgs_and_anns_df["dataset"] = None
            for dataset, df in {"trn": trn_tiles, "val": val_tiles, "tst": tst_tiles}.items():
                valid_imgs_and_anns_df.loc[valid_imgs_and_anns_df["image_id"].isin(df["image_id"]), "dataset"] = dataset
                original_imgs_and_anns_df.loc[original_imgs_and_anns_df["image_id"].isin(df["image_id"]), "dataset"] = dataset
            assert all(valid_imgs_and_anns_df["dataset"].notna()), "Not all images were assigned to a dataset"
            original_imgs_and_anns_df.loc[original_imgs_and_anns_df.dataset.isna(), "dataset"] = "oth"

            logger.info(f"Found {len(trn_tiles)} images in train set, {len(val_tiles)} images in val set and {len(tst_tiles)} images in test set for the {key} dataset.")
            if any(original_imgs_and_anns_df.dataset=='oth'):
                logger.info(f"Found {len(original_imgs_and_anns_df[original_imgs_and_anns_df.dataset=='oth'])} images without validated annotations.")
        del trn_tiles, val_tiles, tst_tiles

    # Iterate through annotations and clip them into tiles
    annotation_id = 0
    excluded_annotations = 0
    image_id = 0
    tot_tiles_with_ann = 0
    tot_tiles_without_ann = 0
    extreme_coordinates_dict = {'min_x': 0, 'max_x': 0, 'min_y': 0, 'max_y': 0}
    gt_tiles_df = pd.DataFrame()
    clipped_annotations_df = pd.DataFrame()
    oth_tiles_df = pd.DataFrame()
    rejected_annotations_df = pd.DataFrame(columns=['id', 'file_name', 'bbox', 'image_name'])
    for aoi, original_imgs_and_anns_df in original_imgs_and_anns_dict.items():
        valid_imgs_and_anns_df = valid_imgs_and_anns_dict[aoi]
        for image in tqdm(original_imgs_and_anns_df.itertuples(), desc=f"Defining tiles and clipping annotations for the {aoi} AOI", total=len(original_imgs_and_anns_df)):

            original_image = os.path.join(IMAGE_DIR[aoi], image.file_name)
            if not os.path.exists(original_image):
                logger.error(f"Image {image.file_name} not found")
                continue
            
            tiles = []
            params = CLIPPING_PARAMS[aoi]
            if aoi == 'SZH':
                if 'lb4' in image.file_name:
                    params = params['lb4']
                else:
                    params = params['else']
            for i in range(params["padding_y"], params["height"] - params["padding_y"] - params["overlap_y"], TILE_SIZE - params["overlap_y"]):
                for j in range(0, params["width"] - params["overlap_x"], TILE_SIZE - params["overlap_x"]):
                    new_filename = os.path.join(OUTPUT_DIR_IMAGES, f"{os.path.basename(image.file_name).rstrip('.jpg')}_{j}_{i}.jpg")
                    tiles.append({
                        "height": TILE_SIZE, "width": TILE_SIZE, "id": image_id, "file_name": new_filename, "dataset": image.dataset,
                        "original_image": original_image, "original_id": image.original_id, "AOI": aoi
                    })
                    image_id += 1

            all_tiles_df = pd.DataFrame(tiles)

            # Clip annotations to tiles
            annotations = []
            if valid_imgs_and_anns_df.empty:
                # Case: inference-only
                tile_annotations_df = pd.DataFrame(columns=['image_id', 'object_id', 'id', 'bbox', 'area', 'category_id', 'iscrowd', 'segmentation'])
            else:
                # Case: training
                border_annotations = 0
                for ann in image.annotations:
                    # Check if annotation is valid
                    rejected_annotation = True
                    validated_annotations = valid_imgs_and_anns_df.loc[valid_imgs_and_anns_df.image_id==image.image_id, 'annotations'].iloc[0]
                    validated_ann = [a for a in validated_annotations if a["id"] == ann["id"]]
                    if len(validated_ann) == 1:
                        # Case: annotation is valid
                        ann = validated_ann[0]
                        rejected_annotation = False
                    elif len(validated_ann) > 1:
                        logger.critical(f"Annotation {ann['id']} is not unique in validated annotations.")
                        sys.exit(1)

                    # Check if annotation is outside the image
                    ann_origin_x, ann_origin_y, ann_width, ann_height = ann["bbox"]
                    if ann_origin_x + ann_width <= 0 or ann_origin_x >= params["width"] or ann_origin_y + ann_height <= 0 or ann_origin_y >= params["height"]:
                        excluded_annotations += 1
                        bbox_coordinates_dict = {'min_x': ann_origin_x, 'max_x': ann_origin_x + ann_width, 'min_y': ann_origin_y, 'max_y': ann_origin_y + ann_height}
                        for key, fct in {'min_x': min, 'max_x': max, 'min_y': min, 'max_y': max}.items():
                            extreme_coordinates_dict[key] = fct(extreme_coordinates_dict[key], bbox_coordinates_dict[key])
                        logger.warning(f"Annotation {ann['id']} is outside the image {image.file_name}. Bbox: {[round(value) for value in ann['bbox']]}.")
                        continue
                    if ann_origin_y > params["height"] - params["padding_y"]:
                        border_annotations += 1
                        continue

                    for tile in tiles:
                        # Get tile coordinates
                        tile_min_x = int(tile["file_name"].split("_")[-2])
                        tile_max_x = tile_min_x + TILE_SIZE
                        tile_min_y = int(tile["file_name"].split("_")[-1].rstrip(".jpg"))
                        tile_max_y = tile_min_y + TILE_SIZE

                        # Check if annotation is outside the tile
                        if ann_origin_x >= tile_max_x or ann_origin_x + ann_width <= tile_min_x or ann_origin_y >= tile_max_y or ann_origin_y + ann_height <= tile_min_y:
                            continue

                        # else, scale coordinates and clip if necessary
                        # bbox
                        x1, new_width = check_bbox_plausibility(ann_origin_x - tile_min_x, ann_width)
                        y1, new_height = check_bbox_plausibility(ann_origin_y - tile_min_y, ann_height)
                        new_coords_tuples = [(x1, y1), (x1 + new_width, y1 + new_height)]
                        if borderline_intersection(new_coords_tuples):
                            border_annotations += 1
                            continue

                        if rejected_annotation:
                            rejected_annotations_df = pd.concat((rejected_annotations_df, pd.DataFrame.from_records([{
                                "id": ann["id"], "file_name": tile["file_name"], "bbox": [x1, y1, new_width, new_height], 'image_name': image.file_name,
                            }])), ignore_index=True)
                            new_coords_tuples = [(x1, y1), (x1 + new_width, y1 + new_height)]
                        else:
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
                            if borderline_intersection(new_coords_tuples):
                                border_annotations += 1
                                continue

                            annotations.append(dict(
                                id=int(annotation_id),
                                object_id = ann["object_id"],
                                image_id=tile["id"],
                                category_id=int(1),  # Currently, single class
                                iscrowd=int(ann["iscrowd"]),
                                bbox=[x1, y1, new_width, new_height],
                                area=misc.segmentation_to_polygon([coords]).area,
                                segmentation=[coords]
                            ))
                            annotation_id += 1

                tile_annotations_df = pd.DataFrame(
                    annotations,
                    columns=['image_id'] if len(annotations) == 0 else annotations[0].keys()
                )
                assert tile_annotations_df.shape[0] + rejected_annotations_df[rejected_annotations_df.image_name == image.file_name].shape[0] + border_annotations + excluded_annotations\
                    >= len(image.annotations), "Missing annotations"
                
            condition_annotations = all_tiles_df["id"].isin(tile_annotations_df["image_id"].unique())

            if RATIO_WO_ANNOTATIONS != 0 and len(annotations) == 0:
                    # TODO: artifacts of old code. Change to respect the ratio of tiles w/o annotation even with tiles from image w/o annotations
                    min_nbr = 1
                    gt_tiles_df = pd.concat((gt_tiles_df, all_tiles_df.sample(n=min_nbr, random_state=SEED)), ignore_index=True)
                    tot_tiles_without_ann += min_nbr
                    selected_tiles = all_tiles_df.sample(n=min_nbr, random_state=SEED).file_name.unique().tolist()

            else: 
                clipped_annotations_df = pd.concat([clipped_annotations_df, tile_annotations_df], ignore_index=True)

                # Separate tiles w/ and w/o annotations
                tiles_with_ann_df = all_tiles_df[condition_annotations]
                tot_tiles_with_ann += tiles_with_ann_df.shape[0]

                if RATIO_WO_ANNOTATIONS != 0:
                    nbr_tiles_without_ann = ceil(len(tiles_with_ann_df) * RATIO_WO_ANNOTATIONS/(1 - RATIO_WO_ANNOTATIONS))
                
                    tiles_without_ann_df = all_tiles_df[~condition_annotations]
                    low_tiles_df = select_low_tiles(tiles_without_ann_df, 1/2)
                    if len(low_tiles_df) >= nbr_tiles_without_ann:
                        added_empty_tiles_df = low_tiles_df.sample(n=nbr_tiles_without_ann, random_state=SEED)
                    else:
                        added_empty_tiles_df = pd.concat([
                            low_tiles_df, 
                            tiles_without_ann_df[~tiles_without_ann_df["file_name"].isin(low_tiles_df["file_name"].unique())].sample(
                                n=nbr_tiles_without_ann-len(low_tiles_df), random_state=SEED
                            )
                        ], ignore_index=True)

                    tot_tiles_without_ann += added_empty_tiles_df.shape[0]

                    gt_tiles_df = pd.concat((gt_tiles_df, tiles_with_ann_df, added_empty_tiles_df), ignore_index=True)
                    selected_tiles = tiles_without_ann_df.file_name.unique().tolist() + added_empty_tiles_df.file_name.unique().tolist()

                else:
                    gt_tiles_df = pd.concat((gt_tiles_df, tiles_with_ann_df), ignore_index=True)
                    selected_tiles = tiles_with_ann_df.file_name.unique().tolist()

            if MAKE_OTHER_DATASET:
                tiles_without_ann_df = all_tiles_df[~(condition_annotations | all_tiles_df["file_name"].isin(selected_tiles))].copy()
                tiles_without_ann_df = select_low_tiles(tiles_without_ann_df, CLIPPING_PARAMS, 1/2)
                oth_tiles_df = pd.concat([oth_tiles_df, tiles_without_ann_df], ignore_index=True)

                del tiles_without_ann_df

            del all_tiles_df, condition_annotations, tile_annotations_df, selected_tiles
                
    logger.info(f"Found {tot_tiles_with_ann} tiles with annotations and {tot_tiles_without_ann} tiles without annotations for training.")
    if excluded_annotations > 0:
        logger.warning(f"{excluded_annotations} annotations were excluded because they were outside of their designated image.")
        logger.warning(f"Most extrem coordinates were:")
        for key, value in extreme_coordinates_dict.items():
            logger.warning(f"- {key.replace('_', ' ')}: {value}")
    # Convert images to tiles
    images_to_tiles_dict = gt_tiles_df.groupby('original_image')['file_name'].apply(list).to_dict()
    gt_tiles_df.drop(columns='original_image', inplace=True)

    achievements = Parallel(n_jobs=10, backend="loky")(delayed(image_to_tiles)(
            image, corresponding_tiles, rejected_annotations_df, tasks_dict=TASKS, overwrite=OVERWRITE_IMAGES
        ) for image, corresponding_tiles in tqdm(images_to_tiles_dict.items(), desc="Converting images to tiles")
    )
    # for image, corresponding_tiles in tqdm(images_to_tiles_dict.items(), desc="Converting images to tiles"):
    #     image_to_tiles(image, corresponding_tiles, rejected_annotations_df, tasks_dict=TASKS, output_dir=OUTPUT_DIR, overwrite=OVERWRITE_IMAGES)
    # del images_to_tiles_dict
    
    if MAKE_OTHER_DATASET:
        logger.info(f"Kept {oth_tiles_df.shape[0]} tiles without annotations in the other dataset.")
        images_to_tiles_dict = oth_tiles_df.groupby('original_image')['file_name'].apply(list).to_dict()
        oth_tiles_df.drop(columns='original_image', inplace=True)

        achievements = Parallel(n_jobs=10, backend="loky")(delayed(image_to_tiles)(
                image, corresponding_tiles, pd.DataFrame(columns=rejected_annotations_df.columns), tasks_dict=TASKS, overwrite=OVERWRITE_IMAGES
            ) for image, corresponding_tiles in tqdm(images_to_tiles_dict.items(), desc="Converting images to tiles")
        )
    del images_to_tiles_dict, achievements

    duplicates = clipped_annotations_df.drop(columns='id').astype({'bbox': str, 'segmentation': str}, copy=True).duplicated()
    if any(duplicates):
        logger.warning(f"Found {duplicates.sum()} duplicated annotations with different ids. Removing them...")
        clipped_annotations_df = clipped_annotations_df[~duplicates].reset_index(drop=True)

    # Create COCO dicts
    dataset_tiles_dict = {
        key: gt_tiles_df[gt_tiles_df["dataset"] == key].drop(columns="dataset").reset_index(drop=True) 
        for key in gt_tiles_df["dataset"].unique()
    }
    if MAKE_OTHER_DATASET:
        dataset_tiles_dict["oth"] = oth_tiles_df.drop(columns="dataset").reset_index(drop=True)

    for dataset in dataset_tiles_dict.keys():
        # Split annotations
        dataset_annotations = clipped_annotations_df[clipped_annotations_df["image_id"].isin(dataset_tiles_dict[dataset]["id"])].copy()
        dataset_annotations = dataset_annotations.astype({"id": int, "category_id": int, "iscrowd": int}, copy=False)
        logger.info(f"Found {len(dataset_annotations)} annotations in the {dataset} dataset.")

        coco_dict = misc.assemble_coco_json(dataset_tiles_dict[dataset], dataset_annotations, CATEGORIES)

        if PREPARE_COCO:
            logger.info(f"Creating COCO file for {dataset} set.")
            with open(os.path.join(COCO_DIR, f"COCO_{dataset}.json"), "w") as fp:
                json.dump(coco_dict, fp, indent=4)
            written_files.append(os.path.join(COCO_DIR, f"COCO_{dataset}.json"))

        if PREPARE_YOLO:
            logger.info(f"Creating COCO file for the annotation transformation to YOLO.")
            dataset_tiles_dict[dataset]["file_name"] = [os.path.basename(f) for f in dataset_tiles_dict[dataset]["file_name"]]
            coco_dict = misc.assemble_coco_json(dataset_tiles_dict[dataset], dataset_annotations, CATEGORIES)

            with open(os.path.join(YOLO_DIR, dataset + '.json'), 'w') as fp:
                json.dump(coco_dict, fp, indent=4)
            written_files.append(os.path.join(YOLO_DIR, dataset + '.json'))

    logger.success("Done! The following files have been created:")
    for file in written_files:
        logger.success(file)
    logger.success(f"In addition, some tiles were written in {', '.join(OUTPUT_DIRS)}.")

    logger.info(f"Done in {round(time() - tic, 2)} seconds.")

        
if __name__ == "__main__":
    
    parser = ArgumentParser(description="This script prepares COCO datasets.")
    parser.add_argument('config_file', type=str, help='a YAML config file')
    args = parser.parse_args()

    main(args.config_file) 