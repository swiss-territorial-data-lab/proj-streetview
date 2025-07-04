import os
import json
from argparse import ArgumentParser
from loguru import logger
from time import time
from tqdm import tqdm
from yaml import load, FullLoader

import pandas as pd
import shapely as shp
from detectron2.data.datasets import load_coco_json, register_coco_instances
from geopandas import GeoDataFrame, GeoSeries, sjoin
from shapely.geometry import MultiPolygon, Polygon
from statistics import median

import utils.misc as misc
from utils.constants import CATEGORIES

logger = misc.format_logger(logger)


def group_annotations(transformed_detections_gdf):
    # Control overlap between detections
    self_join = sjoin(transformed_detections_gdf, transformed_detections_gdf, how='inner')
    valid_self_join = self_join[
        (self_join['id_left'] <= self_join['id_right'])
        & (self_join['image_id_left'] == self_join['image_id_right'])
        & (self_join['dataset_left'] == self_join['dataset_right'])
    ].copy()

    # Do groups because of segmentation on more than two tiles
    groups = misc.make_groups(valid_self_join)
    group_index = {node: i for i, group in enumerate(groups) for node in group}
    valid_self_join = valid_self_join.apply(lambda row: misc.assign_groups(row, group_index), axis=1)

    return valid_self_join


def make_new_annotation(group,groupped_pairs_df, buffer=1):
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


def transform_annotations(tile_name, annotations_df, images_df, images_dir='images', buffer=1, id_field='id', category_field='category_id'):
    name_parts = tile_name.rstrip('.jpg').split('_')
    original_name = os.path.join(images_dir, os.path.basename('_'.join(name_parts[:-2]) + '.jpg'))
    tile_origin_x, tile_origin_y = int(name_parts[-2]), int(name_parts[-1])

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
        buffered_geom = misc.segmentation_to_polygon(ann_segmentation).buffer(buffer)
        ann_geohash = shp.to_wkb(buffered_geom)

        image_id = images_df.loc[images_df.file_name==original_name, 'image_id'].iloc[0]

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
    IMAGE_DIR = cfg['image_dir']
    DETECTIONS_FILES = cfg['detections_files']
    PANOPTIC_COCO_FILES = cfg['panoptic_coco_files']

    SCORE_THRESHOLD = cfg['score_threshold']
    BUFFER = 1

    os.chdir(WORKING_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    logger.info(f"Read detections with a threshold of {SCORE_THRESHOLD} on the confidence score...")
    detections_df = pd.DataFrame()
    for path in DETECTIONS_FILES.values():
        with open(path) as fp:
            detections_df = pd.concat([detections_df, pd.DataFrame.from_records(json.load(fp))], ignore_index=True)

    logger.info(f"Found {len(detections_df)} detections.")
    detections_df = detections_df[detections_df.score >= SCORE_THRESHOLD]
    logger.info(f"{len(detections_df)} detections are left after thresholding on the score.")

    images_df = pd.DataFrame()
    for dataset_key, coco_file in PANOPTIC_COCO_FILES.items():
        register_coco_instances(dataset_key, {}, coco_file, "")

        coco_data = load_coco_json(coco_file, IMAGE_DIR, dataset_key)
        images_df = pd.concat((images_df, pd.DataFrame(coco_data).drop(columns='annotations')), ignore_index=True)

    del coco_data

    transformed_detections= []
    for tile_name in tqdm(detections_df['file_name'].unique(), desc="Tranform detections back to panoptic images"):
        transformed_detections.extend(
            transform_annotations(tile_name, detections_df, images_df, images_dir=IMAGE_DIR, buffer=BUFFER, id_field='det_id', category_field='det_class')
        )

    transformed_detections_gdf = GeoDataFrame(pd.DataFrame.from_records(transformed_detections), geometry='buffered_geometry')

    for dataset in DETECTIONS_FILES.keys():
        logger.info(f'Working on the {dataset} dataset...')
        subset_transformed_detections_gdf = transformed_detections_gdf[transformed_detections_gdf.dataset==dataset].copy()

        logger.info('Groupping overlapping detections...')
        groupped_pairs_df = group_annotations(subset_transformed_detections_gdf)

        merged_detections = []
        for group in tqdm(groupped_pairs_df.group_id.unique(), desc="Merge detections in groups"):
            merged_detections.append(make_new_annotation(group, groupped_pairs_df, buffer=BUFFER))
        logger.info(f"{len(merged_detections)} detections are left after merging.")

        logger.info("Transforming detections to COCO format...")
        subset_images_df = images_df[images_df.image_id.isin(subset_transformed_detections_gdf.image_id.unique())].copy()
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