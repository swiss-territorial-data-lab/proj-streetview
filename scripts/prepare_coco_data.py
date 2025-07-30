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
from math import ceil

from utils.constants import CATEGORIES, TILE_SIZE
from utils.misc import assemble_coco_json, format_logger, read_coco_dataset, segmentation_to_polygon

logger = format_logger(logger)

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


def image_to_tiles(image, corresponding_tiles, rejected_annotations_df, image_height, image_width, image_dir, tasks_dict, overwrite=False):
    """
    Processes an image by dividing it into tiles, applying masks on pixels corresponding to rejected annotations, and saving the tiles.

    Args:
        image (str): The filename of the image to process.
        corresponding_tiles (list): List of tile names that represent the sub-regions of the image.
        rejected_annotations_df (DataFrame): DataFrame containing annotations that need to be masked on the tiles.
        image_height (int): The pixel height of the image.
        image_width (int): The pixel width of the image.
        image_dir (str): Directory path where the image is located.
        tasks_dict (dict): A dictionary defining the tasks to prepare data for, including subfolder paths.
        overwrite (bool, optional): Flag indicating whether existing files should be overwritten. Defaults to False.

    Returns:
        dict: A dictionary with the image name as the key and a list of booleans indicating the success of saving each tile.
    """

    prepare_coco = tasks_dict['coco']['prepare_data'] if 'coco' in tasks_dict.keys() else False
    prepare_yolo = tasks_dict['yolo']['prepare_data'] if 'yolo' in tasks_dict.keys() else False
    output_dirs = [tasks_dict[task]['subfolder'] for task in tasks_dict.keys() if tasks_dict[task]['prepare_data']]
    achieved = {image: []}

    img = cv2.imread(os.path.join(image_dir, image))

    h, w = img.shape[:2]
    assert h == image_height, f"Image height not {image_height} px"
    assert w == image_width, f"Image width not {image_width} px"

    for tile_name in corresponding_tiles:
        files_exist = all([os.path.exists(os.path.join(output_dir, tile_name)) for output_dir in output_dirs])
        if not files_exist or overwrite:
            i = int(tile_name.split("_")[-1].rstrip(".jpg"))
            j = int(tile_name.split("_")[-2])
            tile = np.zeros((TILE_SIZE, TILE_SIZE, 3), dtype=np.uint8)
            tile[:] = img[i:i+TILE_SIZE, j:j+TILE_SIZE]     # deep copy
            assert tile.shape[0] == TILE_SIZE and tile.shape[1] == TILE_SIZE, f"Tile shape not {TILE_SIZE} x {TILE_SIZE} px"

            # Draw a black mask on reject annotations
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
                
            assert annotations_to_mask_df.empty or np.any(tile != img[i:i+TILE_SIZE, j:j+TILE_SIZE]), "Mask not applied"
        
            if prepare_coco and prepare_yolo:
                tile_path = os.path.join(tasks_dict['coco']['subfolder'], tile_name)
                achieved_coco = cv2.imwrite(tile_path, tile)

                dest_path = os.path.join(tasks_dict['yolo']['subfolder'], os.path.basename(tile_name))
                os.link(tile_path, dest_path)

                achieved[image].append(achieved_coco and os.path.exists(dest_path))

            elif prepare_coco:
                tile_path = os.path.join(tasks_dict['coco']['subfolder'], tile_name)
                achieved[image].append(cv2.imwrite(tile_path, tile))

            elif prepare_yolo:
                tile_path = os.path.join(tasks_dict['yolo']['subfolder'], os.path.basename(tile_name))
                achieved[image].append(cv2.imwrite(tile_path, tile))

    return achieved

def write_image(image, image_name, dir_name, overwrite):
    filepath = os.path.join(dir_name, image_name)
    if not os.path.exists(filepath) or overwrite:
        cv2.imwrite(filepath, image)

def select_low_tiles(tiles_df, excluded_height_ratio=2/3):
    """
    Select tiles that are above a certain height ratio.

    Args:
        tiles_df (DataFrame): A DataFrame containing the tiles.
        excluded_height_ratio (float, optional): The height ratio above which tiles should be excluded. Defaults to 2/3.

    Returns:
        DataFrame: A DataFrame containing the selected tiles.
    """
    _tiles_df = tiles_df.copy()
    _tiles_df["row_level"] = _tiles_df["file_name"].apply(lambda x: int(x.split("_")[-1].rstrip(".jpg")))
    max_height = _tiles_df["row_level"].max()
    low_tiles_df = _tiles_df[_tiles_df["row_level"] > max_height*excluded_height_ratio]

    return low_tiles_df


def main(cfg_file_path):
    
    tic = time()
    logger.info('Starting...')

    logger.info(f"Using {cfg_file_path} as config file.")

    with open(cfg_file_path) as fp:
        cfg = load(fp, Loader=FullLoader)[os.path.basename(__file__)]

    WORKING_DIR = cfg['working_directory']
    IMAGE_DIR = cfg['image_dir']

    ORIGINAL_COCO_FILES_DICT = cfg['original_COCO_files']
    VALIDATED_COCO_FILES_DICT = cfg['validated_COCO_files']
    RATIO_WO_ANNOTATIONS = cfg['ratio_wo_annotations']
    SEED = cfg['seed']
    OVERWRITE_IMAGES = cfg['overwrite_images']

    TASKS = cfg['tasks']
    MAKE_OTHER_DATASET = TASKS['make_other_dataset']
    TASKS.pop('make_other_dataset')
    PREPARE_COCO = TASKS['coco']['prepare_data'] if 'coco' in TASKS.keys() else False
    PREPARE_YOLO = TASKS['yolo']['prepare_data'] if 'yolo' in TASKS.keys() else False

    IMAGE_HEIGHT = 4000
    IMAGE_WIDTH = 8000
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
        COCO_DIR = TASKS['coco']['subfolder']
        os.makedirs(COCO_DIR, exist_ok=True)
        # Subfolder for COCO images
        os.makedirs(os.path.join(COCO_DIR, OUTPUT_DIR_IMAGES), exist_ok=True)
        OUTPUT_DIRS.append(os.path.join(COCO_DIR, OUTPUT_DIR_IMAGES))
    if PREPARE_YOLO:
        # YOLO conversion requires tiles to be saved in the same folder as COCO files
        YOLO_DIR = TASKS['yolo']['subfolder']
        os.makedirs(YOLO_DIR, exist_ok=True)
        OUTPUT_DIRS.append(os.path.join(YOLO_DIR))

    logger.info(f"Read COCO files...")
    # Read full COCO dataset
    original_imgs_and_anns_df = pd.DataFrame()
    for _, coco_file in ORIGINAL_COCO_FILES_DICT.items():
        images_df = read_coco_dataset(coco_file)
        original_imgs_and_anns_df = pd.concat((original_imgs_and_anns_df, images_df), ignore_index=True)

    # Read validated COCO dataset
    valid_imgs_and_anns_df = pd.DataFrame()
    for _, coco_file in VALIDATED_COCO_FILES_DICT.items():
        images_df = read_coco_dataset(coco_file)
        valid_imgs_and_anns_df = pd.concat((valid_imgs_and_anns_df, images_df), ignore_index=True)

    if DEBUG:
        logger.info("Debug mode activated. Only first 100 images are processed.")
        original_imgs_and_anns_df = original_imgs_and_anns_df.head(100)
        valid_imgs_and_anns_df = valid_imgs_and_anns_df.head(150)

    logger.info(f"Found {len(valid_imgs_and_anns_df)} images for validated annotations.")

    logger.info("Splitting images into train, val and test sets based on ratio 70% / 15% / 15%...")
    trn_tiles = valid_imgs_and_anns_df.sample(frac=0.7, random_state=SEED)
    val_tiles = valid_imgs_and_anns_df[~valid_imgs_and_anns_df["image_id"].isin(trn_tiles["image_id"])].sample(frac=0.5, random_state=SEED)
    tst_tiles = valid_imgs_and_anns_df[~valid_imgs_and_anns_df["image_id"].isin(trn_tiles["image_id"].to_list() + val_tiles["image_id"].to_list())]

    # Map dataset on the images
    valid_imgs_and_anns_df["dataset"] = None
    for dataset, df in {"trn": trn_tiles, "val": val_tiles, "tst": tst_tiles}.items():
        valid_imgs_and_anns_df.loc[valid_imgs_and_anns_df["image_id"].isin(df["image_id"]), "dataset"] = dataset
        original_imgs_and_anns_df.loc[original_imgs_and_anns_df["image_id"].isin(df["image_id"]), "dataset"] = dataset
    assert all(valid_imgs_and_anns_df["dataset"].notna()), "Not all images were assigned to a dataset"
    original_imgs_and_anns_df.loc[original_imgs_and_anns_df.dataset.isna(), "dataset"] = "oth"

    logger.info(f"Found {len(trn_tiles)} images in train set, {len(val_tiles)} images in val set and {len(tst_tiles)} images in test set.")
    del trn_tiles, val_tiles, tst_tiles

    # Iterate through annotations and clip them into tiles
    image_id = 0
    annotation_id = 0
    gt_tiles_df = pd.DataFrame()
    oth_tiles_df = pd.DataFrame()
    clipped_annotations_df = pd.DataFrame()
    rejected_annotations_df = pd.DataFrame(columns=['id', 'file_name', 'bbox'])
    tot_tiles_with_ann = 0
    tot_tiles_without_ann = 0
    for image in tqdm(original_imgs_and_anns_df.itertuples(), desc="Defining tiles and clipping annotations to tiles", total=len(original_imgs_and_anns_df)):
            
        if not os.path.exists(os.path.join(IMAGE_DIR, image.file_name)):
            logger.error(f"Image {image.file_name} not found")
            continue

        tiles = []
        for i in range(PADDING_Y, IMAGE_HEIGHT - PADDING_Y, TILE_SIZE - OVERLAP_Y):
            for j in range(0, IMAGE_WIDTH - OVERLAP_X, TILE_SIZE - OVERLAP_X):
                new_filename = os.path.join(OUTPUT_DIR_IMAGES, f"{os.path.basename(image.file_name).rstrip('.jpg')}_{j}_{i}.jpg")
                tiles.append({
                    "height": TILE_SIZE, "width": TILE_SIZE, "id": image_id, "file_name": new_filename, "dataset": image.dataset, "original_image": image.file_name
                })
                image_id += 1

        all_tiles_df = pd.DataFrame(tiles)

        # Clip annotations to tiles
        annotations = []
        for ann in image.annotations:
            # Check if annotation is valid
            rejected_annotation = True
            validated_annotations = valid_imgs_and_anns_df.loc[valid_imgs_and_anns_df.image_id==image.image_id, 'annotations'].iloc[0]
            validated_ann = [a for a in validated_annotations if a["id"] == ann["id"]]
            if len(validated_ann) == 1:
                ann = validated_ann[0]
                rejected_annotation = False
            elif len(validated_ann) > 1:
                logger.critical(f"Annotation {ann['id']} is not unique in validated annotations.")
                sys.exit(1)

            for tile in tiles:
                # Get tile coordinates
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

                if rejected_annotation:
                    rejected_annotations_df = pd.concat((rejected_annotations_df, pd.DataFrame.from_records([{
                        "id": ann["id"], "file_name": tile["file_name"], "bbox": [x1, y1, new_width, new_height]
                    }])), ignore_index=True)
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
                    if all(value[0] <= TILE_SIZE * 0.02 or value[0] >= TILE_SIZE * 0.98 for value in new_coords_tuples) or all(value[1] <=10 or value[1] >= 500 for value in new_coords_tuples):
                        continue

                annotations.append(dict(
                    id=int(annotation_id),
                    image_id=tile["id"],
                    category_id=int(1),  # Currently, single class
                    iscrowd=int(ann["iscrowd"]),
                    bbox=[x1, y1, new_width, new_height],
                    area=segmentation_to_polygon([coords]).area,
                    segmentation=[coords]
                ))
                annotation_id += 1

        tile_annotations_df = pd.DataFrame(
            annotations,
            columns=['image_id'] if len(annotations) == 0 else annotations[0].keys()
        )
        condition_annotations = all_tiles_df["id"].isin(tile_annotations_df["image_id"].unique())

        if RATIO_WO_ANNOTATIONS != 0 and len(annotations) == 0:
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
                low_tiles_df = select_low_tiles(tiles_without_ann_df, 2/3)
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
            tiles_with_ann_df = select_low_tiles(tiles_without_ann_df, 3/4)
            oth_tiles_df = pd.concat([oth_tiles_df, tiles_without_ann_df], ignore_index=True)

            del tiles_without_ann_df

        del all_tiles_df, condition_annotations, tile_annotations_df, selected_tiles
                
    logger.info(f"Found {tot_tiles_with_ann} tiles with annotations and kept {tot_tiles_without_ann} tiles without annotations in the training datasets.")
    # Convert images to tiles
    images_to_tiles_dict = gt_tiles_df.groupby('original_image')['file_name'].apply(list).to_dict()
    gt_tiles_df.drop(columns='original_image', inplace=True)

    _ = Parallel(n_jobs=10, backend="loky")(delayed(image_to_tiles)(
            image, corresponding_tiles, rejected_annotations_df, IMAGE_HEIGHT, IMAGE_WIDTH, image_dir=IMAGE_DIR, tasks_dict=TASKS, overwrite=OVERWRITE_IMAGES
        ) for image, corresponding_tiles in tqdm(images_to_tiles_dict.items(), desc="Converting images to tiles")
    )
    # for image, corresponding_tiles in tqdm(images_to_tiles_dict.items(), desc="Converting images to tiles"):
    #     image_to_tiles(image, corresponding_tiles, rejected_annotations_df, IMAGE_HEIGHT, IMAGE_WIDTH, image_dir=IMAGE_DIR, tasks_dict=TASKS, overwrite=OVERWRITE_IMAGES)
    # del images_to_tiles_dict
    
    if MAKE_OTHER_DATASET:
        logger.info(f"Kept {oth_tiles_df.shape[0]} tiles without annotations in the other dataset.")
        images_to_tiles_dict = oth_tiles_df.groupby('original_image')['file_name'].apply(list).to_dict()
        oth_tiles_df.drop(columns='original_image', inplace=True)

        _ = Parallel(n_jobs=10, backend="loky")(delayed(image_to_tiles)(
                image, corresponding_tiles, pd.DataFrame(columns=rejected_annotations_df.columns), IMAGE_HEIGHT, IMAGE_WIDTH, image_dir=IMAGE_DIR, tasks_dict=TASKS, overwrite=OVERWRITE_IMAGES
            ) for image, corresponding_tiles in tqdm(images_to_tiles_dict.items(), desc="Converting images to tiles")
        )

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
    logger.success(f"In addition, some tiles were written in {', '.join(OUTPUT_DIRS)}.")

    logger.info(f"Done in {round(time() - tic, 2)} seconds.")

        
if __name__ == "__main__":
    
    parser = ArgumentParser(description="This script prepares COCO datasets.")
    parser.add_argument('config_file', type=str, help='a YAML config file')
    args = parser.parse_args()

    main(args.config_file) 