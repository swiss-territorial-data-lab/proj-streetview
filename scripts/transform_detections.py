import os
import json
from argparse import ArgumentParser
from loguru import logger
from time import time
from tqdm import tqdm
from yaml import load, FullLoader

import pandas as pd
import shapely as shp
from geopandas import GeoDataFrame, GeoSeries, sjoin
from shapely.geometry import MultiPolygon, Polygon
from statistics import median

import utils.misc as misc
from utils.constants import CATEGORIES, IMAGE_DIR

logger = misc.format_logger(logger)


def group_annotations(transformed_detections_gdf, verbose=False):
    """
    Groups overlapping annotations in the transformed detections GeoDataFrame.

    This function performs a spatial self-join on the `transformed_detections_gdf` to find 
    overlapping annotation pairs that belong to the same image and dataset. It then groups 
    these overlapping annotations using a graph-based approach, assigning a unique group 
    ID to each connected component of overlapping annotations.

    Args:
        transformed_detections_gdf (GeoDataFrame): A GeoDataFrame containing transformed 
        detections with geometry information.
        verbose (bool, optional): Whether to display progress bars. Defaults to False.

    Returns:
        DataFrame: A DataFrame with the same columns as `transformed_detections_gdf`, 
        but with an additional column 'group_id' indicating the group number each 
        annotation belongs to.
    """

    # Find overlapping pairs
    overlapping_dets_gdf = GeoDataFrame()
    for image_id in tqdm(transformed_detections_gdf.image_id.unique(), desc="Find overlapping pairs", disable=(not verbose)):
        subset_gdf = transformed_detections_gdf[transformed_detections_gdf.image_id==image_id]
        self_join = sjoin(subset_gdf, subset_gdf, how='inner')
        overlapping_dets_gdf = pd.concat([
            overlapping_dets_gdf,
            self_join[
                (self_join['id_left'] <= self_join['id_right'])
                & (self_join['dataset_left'] == self_join['dataset_right'])
            ]
        ], ignore_index=True)

    # Do groups because of segmentation on more than two tiles
    groups = misc.make_groups(overlapping_dets_gdf)
    group_index = {node: i for i, group in enumerate(groups) for node in group}
    overlapping_dets_gdf = overlapping_dets_gdf.apply(lambda row: misc.assign_groups(row, group_index), axis=1)

    return overlapping_dets_gdf


def make_new_annotation(group,groupped_pairs_df, buffer=1):
    """
    Creates a new annotation based on a group of overlapping detections.

    Args:
        group (int): The group number of the overlapping detections.
        groupped_pairs_df (DataFrame): A DataFrame containing the overlapping pairs
            of detections, with group number assigned to each pair.
        buffer (int, optional): The buffer size to subtract from the new geometry.
            Defaults to 1.

    Returns:
        dict: A dictionary containing the new annotation information.
    """
    group_dets = groupped_pairs_df[groupped_pairs_df.group_id==group].copy()

    # Keep lowest id, median score. Calculate new segmentation, area and bbox
    new_geometry = shp.unary_union(
        pd.concat([group_dets.buffered_geometry, GeoSeries(group_dets.geohash_right.apply(shp.from_wkb))]).drop_duplicates()
    ).buffer(-buffer)
    new_segmentation = polygon_to_segmentation(new_geometry)
    bbox = new_geometry.bounds
    
    ann_dict = {
        'id': int(group_dets.id_left.min()),
        'image_id': int(group_dets.image_id_left.iloc[0]),
        'category_id': int(group_dets.category_id_left.iloc[0]),
        'dataset': group_dets.dataset_left.iloc[0],
        'bbox': [bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]],
        'segmentation': new_segmentation,
        'area': new_geometry.area,
    }

    if 'score_left' in group_dets.columns:
        # Todo: ensure each score appear only once
        ann_dict['score'] = median(group_dets.score_left.tolist() + group_dets.score_right.tolist())

    return ann_dict


def polygon_to_segmentation(multipolygon):
    """
    Convert a shapely Polygon or MultiPolygon into a COCO-style segmentation.

    Args:
        polygon (Polygon or MultiPolygon): A shapely geometry object representing the polygon(s).

    Returns:
        list: A list of lists where each sublist contains the x and y coordinates of the polygon's exterior in 
              a flattened format suitable for COCO segmentation.
    """

    segmentation = []
    polygon_coords = []
    if isinstance(multipolygon, Polygon):
        multipolygon = MultiPolygon([multipolygon])

    for poly in multipolygon.geoms:
        exterior_coords = poly.exterior.coords.xy # Missing inner rings if polygon has holes
        for coord_index in range(len(exterior_coords[0])):
            polygon_coords.append(exterior_coords[0][coord_index])
            polygon_coords.append(exterior_coords[1][coord_index])
    segmentation.append(polygon_coords)
    
    return segmentation


def read_image_info(coco_file_path_dict, id_correspondence_df):
    """
    Reads image information from multiple COCO JSON files and merges it with an ID correspondence DataFrame.

    Args:
        coco_file_path_dict (dict): A dictionary where keys are dataset identifiers and values are paths to
                                    COCO JSON files containing image data.
        id_correspondence_df (DataFrame): A DataFrame containing ID correspondence information with columns
                                          for 'dataset', 'original_id', and 'image_id'.

    Returns:
        DataFrame: A DataFrame containing combined image information from all specified COCO JSON files,
                   including a 'basename' column with the base filenames.
    """

    images_df = pd.DataFrame()
    for dataset_key, coco_file in coco_file_path_dict.items():
        with open(coco_file) as fp:
            coco_data = json.load(fp)['images']

        tmp_df = pd.DataFrame(coco_data)
        tmp_df = tmp_df.merge(
            id_correspondence_df[id_correspondence_df.dataset==dataset_key], 
            how='left', left_on='id', right_on='original_id'
        ).drop(columns=['original_id','id'])
        images_df = pd.concat((images_df, tmp_df), ignore_index=True)

    images_df['basename'] = images_df.file_name.apply(lambda x: os.path.basename(x))

    return images_df


def transform_annotations(tile_name, annotations_df, images_df, buffer=1, id_field='id', category_field='category_id'):
    """
    Transform COCO annotations on a tile to their original image and pixel coordinates.

    Args:
        tile_name (str): The name of the tile, including the directory and extension.
        annotations_df (DataFrame): A DataFrame containing the COCO annotations for the tile.
        images_df (DataFrame): A DataFrame containing information about the original images.
        buffer (int, optional): The amount of pixels to buffer the geometry of each annotation. Defaults to 1.
        id_field (str, optional): The name of the column in annotations_df containing the annotation ID. Defaults to 'id'.
        category_field (str, optional): The name of the column in annotations_df containing the category ID. Defaults to 'category_id'.

    Returns:
        list: A list of dictionaries, each representing an annotation on the original image.

    Raises:
        ValueError: If no image is found with the same name as the tile.
        ValueError: If multiple images are found with the same name.
    """
    
    name_parts = tile_name.rstrip('.jpg').split('_')
    original_name = os.path.basename('_'.join(name_parts[:-2]) + '.jpg')
    tile_origin_x, tile_origin_y = int(name_parts[-2]), int(name_parts[-1])

    corresponding_images = images_df.loc[images_df.basename==original_name, 'image_id']
    if len(corresponding_images) == 1:
        image_id = corresponding_images.iloc[0]
    elif len(corresponding_images) > 1:
        raise ValueError(f"Multiple images with the same name: {original_name}")
    else:
        raise ValueError(f"No image with the name: {original_name}")
    annotations_on_tiles_df = annotations_df[annotations_df.file_name==tile_name].copy()
    annotations_on_tiles_list = []
    for ann in annotations_on_tiles_df.itertuples():

        ann_segmentation = []
        for poly in ann.segmentation:
            poly_coordinates = []
            for coor_id in range(0, len(poly), 2):
                poly_coordinates.append(poly[coor_id] + tile_origin_x)
                poly_coordinates.append(poly[coor_id + 1] + tile_origin_y)
            ann_segmentation.append(poly_coordinates)
        # Buffer geometry to facilitate overlap in the next step
        buffered_geom = misc.segmentation_to_polygon(ann_segmentation).buffer(buffer)
        ann_geohash = shp.to_wkb(buffered_geom)

        annotations_on_tiles_list.append({
            'id': getattr(ann, id_field),
            'image_id': image_id,
            'category_id': getattr(ann, category_field),
            'dataset': ann.dataset,
            'segmentation': ann_segmentation,
            'buffered_geometry': buffered_geom,
            'geohash': ann_geohash
        })

        if 'score' in annotations_on_tiles_df.columns:
            annotations_on_tiles_list[-1]['score'] = round(ann.score, 3)

    return annotations_on_tiles_list


def main(cfg_file_path):

    tic = time()
    logger.info('Starting...')

    logger.info(f"Using {cfg_file_path} as config file.")

    with open(cfg_file_path) as fp:
        cfg = load(fp, Loader=FullLoader)[os.path.basename(__file__)]

    WORKING_DIR = cfg['working_directory']
    OUTPUT_DIR = cfg['output_folder']
    DETECTIONS_FILES = cfg['detections_files']
    PANOPTIC_COCO_FILES = cfg['panoptic_coco_files']
    ID_CORRESPONDENCE = cfg['id_correspondence']

    SCORE_THRESHOLD = cfg['score_threshold']
    BUFFER = 1

    os.chdir(WORKING_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    logger.info(f"Read detections with a threshold of {SCORE_THRESHOLD} on the confidence score...")
    detections_df = pd.DataFrame()
    for dataset_key, path in DETECTIONS_FILES.items():
        with open(path) as fp:
            dets = pd.DataFrame.from_records(json.load(fp))
        dets['dataset'] = dataset_key
        detections_df = pd.concat([detections_df, dets], ignore_index=True)

    logger.info(f"Found {len(detections_df)} detections.")
    detections_df = detections_df[detections_df.score >= SCORE_THRESHOLD]
    logger.info(f"{len(detections_df)} detections are left after thresholding on the score.")

    id_correspondence_df = pd.read_csv(ID_CORRESPONDENCE)
    images_df = read_image_info(PANOPTIC_COCO_FILES, id_correspondence_df)

    transformed_detections= []
    for tile_name in tqdm(detections_df['file_name'].unique(), desc="Tranform detections back to panoptic images"):
        transformed_detections.extend(
            transform_annotations(tile_name, detections_df, images_df, buffer=BUFFER, id_field='det_id', category_field='det_class')
        )

    transformed_detections_gdf = GeoDataFrame(pd.DataFrame.from_records(transformed_detections), geometry='buffered_geometry')
    transformed_detections_gdf = transformed_detections_gdf[~transformed_detections_gdf.geometry.is_empty]

    for dataset in DETECTIONS_FILES.keys():
        logger.info(f'Working on the {dataset} dataset...')
        subset_transformed_detections_gdf = transformed_detections_gdf[transformed_detections_gdf.dataset==dataset].copy()

        logger.info('Groupping overlapping detections...')
        groupped_pairs_df = group_annotations(subset_transformed_detections_gdf, verbose=True)

        merged_detections = []
        for group in tqdm(groupped_pairs_df.group_id.unique(), desc="Merge detections in groups"):
            merged_detections.append(make_new_annotation(group, groupped_pairs_df, buffer=BUFFER))
        logger.info(f"{len(merged_detections)} detections are left after merging.")

        logger.info("Transforming detections to COCO format...")
        subset_images_df = images_df[images_df.image_id.isin(subset_transformed_detections_gdf.image_id.unique())].rename(columns={'image_id': 'id'})
        CATEGORIES[0]['id'] = 0 # COCO usually starts with 1, but detectron2 starts with 0
        coco_dict = misc.assemble_coco_json(subset_images_df, merged_detections, CATEGORIES)

        # Save to coco json
        filepath = os.path.join(OUTPUT_DIR, f'{dataset}_COCO_panoptic_detections.json')
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