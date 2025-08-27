import os
from argparse import ArgumentParser
from loguru import logger
from time import time
from yaml import FullLoader, load

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio as rio
import seaborn as sns
from shapely.geometry import MultiPolygon, Polygon

import utils.metrics as metrics
from utils.constants import CATEGORIES
from utils.misc import format_logger, grid_over_tile

logger = format_logger(logger)

def format_and_save_tagged_results(tagged_res_gdf, prefix, output_dir):
    tagged_res_gdf = gpd.GeoDataFrame(tagged_res_gdf, geometry='geometry', crs='epsg:2056')
    tagged_res_gdf.drop(columns=['det_class', 'label_class'], inplace=True)
    tagged_res_gdf.rename(columns={'label_id': 'cadaster_id'}, inplace=True)

    # Map tags into values understandable by the user
    new_tag = {'TP': 'confirmed', 'FP': 'new detection', 'FN': 'missing'}
    tagged_res_gdf.loc[:, 'tag'] = tagged_res_gdf['tag'].map(new_tag)

    filepath = os.path.join(output_dir, prefix + 'tagged_detections.gpkg')
    tagged_res_gdf.to_file(filepath)
    
    return filepath


def read_files(files_dict, aoi_poly):
    obj_dict = {}
    for dataset_key, filepath in files_dict.items():
        tmp_gdf = gpd.read_file(filepath)
        obj_dict[dataset_key] = tmp_gdf[tmp_gdf.intersects(aoi_poly)]
    # We suppose geom type stable across datasets
    geom_type = [gdf for gdf in obj_dict.values()][0].geom_type.iloc[0]

    return obj_dict, geom_type

parser = ArgumentParser(description="This script compares the existing pipe cadater to the data produced by the algorithm.")
parser.add_argument('config_file', type=str, help='a YAML config file')
args = parser.parse_args()

cfg_file_path = os.path.join(os.path.dirname(__file__), args.config_file)
tic = time()
logger.info('Starting...')

logger.info(f"Using {cfg_file_path} as config file.")

with open(cfg_file_path) as fp:
    cfg = load(fp, Loader=FullLoader)[os.path.basename(__file__)]

WORKING_DIR = cfg['working_directory']
OUTPUT_DIR = cfg['output_dir']

AOI_FILES = cfg['aoi_files']
DETECTION_FILES = cfg['detection_files']
CADASTER_MANHOLE_FILES = cfg['cadaster_manhole_files']
ID_OBJ_CADASTER = cfg['id_obj_cadaster'] if 'id_obj_cadaster' in cfg.keys() else None

IOU_THRESHOLD = cfg['iou_threshold']
SAVE_SEPARATE_DATASETS = cfg['save_separate_datasets']

os.chdir(WORKING_DIR)
logger.info(f'Working directory set to {WORKING_DIR}.')
os.makedirs(OUTPUT_DIR, exist_ok=True)
written_files = []

logger.info("Reading data...")

aoi_gdf = gpd.GeoDataFrame()
for dataset_key, filepath in AOI_FILES.items():
    tmp_gdf = gpd.read_file(filepath)
    if tmp_gdf.geom_type.iloc[0] == 'Point':
        tmp_gdf.loc[:, 'geometry'] = tmp_gdf.buffer(20)
    aoi_gdf = pd.concat([aoi_gdf, tmp_gdf], ignore_index=True)
aoi_poly = aoi_gdf.union_all()

detections_dict, geom_type_dets = read_files(DETECTION_FILES, aoi_poly)

manholes_dict, geom_type_cadaster = read_files(CADASTER_MANHOLE_FILES, aoi_poly)

assert detections_dict.keys() == manholes_dict.keys(), "Datasets must be the same for detections and pipe cadaster."

logger.info("Found the following datasets:")
for dataset_key in detections_dict.keys():
    logger.info(f"- {dataset_key}")
    logger.info(f"    - {detections_dict[dataset_key].shape[0]} detections")
    logger.info(f"    - {manholes_dict[dataset_key].shape[0]} manholes")

logger.info(f"Detections are {geom_type_dets}s and manholes are {geom_type_cadaster}s.")
logger.info(f"Points are transformed to polygons with a buffer of 30 cm.")

for geom_type, dataset_dict in [(geom_type_dets, detections_dict), (geom_type_cadaster, manholes_dict)]:
    if geom_type == 'Point':
        for gdf in dataset_dict.values():
            gdf.loc[:, 'geometry'] = gdf.buffer(0.3)


# Define class
id_classes = [0]
categories_info_df = pd.DataFrame(CATEGORIES[0], index=[0]).rename(columns={'id': 'label_class', 'name': 'category'})

logger.info("Comparing detections and manholes...")
logger.info("The cadaster data are acting as ground truth.")

global_metrics_dict = {'dataset': [], 'TP': [], 'FP': [], 'FN': [], 'precision': [], 'recall': [], 'f1': []}    
tagged_res_gdf_dict = {}
for dataset_key in detections_dict.keys():
    detections_gdf = detections_dict[dataset_key].copy()
    cadaster_manholes_gdf = manholes_dict[dataset_key].copy()

    if 'det_id' not in detections_gdf.columns:
        detections_gdf['det_id'] = detections_gdf.index
    detections_gdf['det_class'] = 0
    if ID_OBJ_CADASTER:
        cadaster_manholes_gdf['label_id'] = ID_OBJ_CADASTER
    else:
        cadaster_manholes_gdf['label_id'] = cadaster_manholes_gdf.index
    cadaster_manholes_gdf['label_class'] = 1

    tagged_gdf_dict = metrics.get_fractional_sets(
        detections_gdf, 
        cadaster_manholes_gdf,
        dataset_key,
        IOU_THRESHOLD
    )

    tp_k, fp_k, fn_k, p_k, r_k, precision, recall, f1 = metrics.get_metrics(id_classes=id_classes, **tagged_gdf_dict)
    global_metrics_dict['dataset'].append(dataset_key)
    for key, value in {'TP': tp_k[0], 'FP': fp_k[0], 'FN': fn_k[0], 'precision': precision, 'recall': recall, 'f1': f1}.items():
        global_metrics_dict[key].append(value)
    logger.info(f'Dataset = {dataset_key} => precision = {precision:.3f}, recall = {recall:.3f}, f1 = {f1:.3f}')

    tagged_res_gdf_dict[dataset_key] = pd.concat(tagged_gdf_dict.values())

logger.info('Save results...')
# Format tagged results
if SAVE_SEPARATE_DATASETS:
    for dataset_key in detections_dict.keys():
        written_files.append(
            format_and_save_tagged_results(tagged_res_gdf_dict[dataset_key], prefix=f'{dataset_key}_', output_dir=OUTPUT_DIR)
        )
else:
    written_files.append(
        format_and_save_tagged_results(pd.concat(tagged_res_gdf_dict.values()), prefix='', output_dir=OUTPUT_DIR)
    )

# Format global metrics
global_metrics_df = pd.DataFrame(global_metrics_dict)

filepath = os.path.join(OUTPUT_DIR, 'global_metrics.csv')
global_metrics_df.to_csv(filepath, index=False)
written_files.append(filepath)

logger.info("Calculate density of problematic points on a grid...")
TILE_SIZE = 100
full_grid_gdf = gpd.GeoDataFrame()
if aoi_poly.geom_type == 'Polygon':
    aoi_parts = [aoi_poly]
else:
    aoi_parts = aoi_poly.geoms
for zone in aoi_parts:
    tile_origin = zone.bounds[0], zone.bounds[1]
    tile_size = (
        (zone.bounds[2] - zone.bounds[0])/TILE_SIZE, 
        (zone.bounds[3] - zone.bounds[1])/TILE_SIZE
    )
    grid_gdf = grid_over_tile(tile_size, tile_origin, pixel_size_x=TILE_SIZE, grid_width=1, grid_height=1, test_shape=zone)
    full_grid_gdf = pd.concat([full_grid_gdf, grid_gdf])

tmp_df = gpd.GeoDataFrame(pd.concat(tagged_res_gdf_dict.values()))
problematic_points_gdf = gpd.GeoDataFrame(
    tmp_df[tmp_df['tag'].isin(['FN', 'FP'])],
    geometry=tmp_df.loc[tmp_df['tag'].isin(['FN', 'FP'])].centroid,
    columns=['det_id', 'label_id', 'tag', 'geometry'],
    crs='EPSG:2056'
)

points_on_grid_gdf = gpd.sjoin(full_grid_gdf, problematic_points_gdf, how='inner')
point_count_df = points_on_grid_gdf.groupby(['id', 'tag']).size().reset_index(name='count')
point_count_gdf = full_grid_gdf.merge(point_count_df, how='right', on=['id'])

logger.info('Save result...')
filepath = os.path.join(OUTPUT_DIR, 'heatmap_grid.gpkg')
point_count_gdf.to_file(filepath)
written_files.append(filepath)

logger.info('Produce a KDE plot of problematic points...')
# cf. https://towardsdatascience.com/from-kernel-density-estimation-to-spatial-analysis-in-python-64ddcdb6bc9b/
levels = [0.2, 0.3, 0.4, 0.45, 0.475, 0.5, 0.525, 0.55, 0.575, 0.6, 0.625, 0.65, 0.675, 0.7, 0.725, 0.75, 0.8, 0.9,1]
f, ax = plt.subplots(ncols=1, figsize=(20, 8))
# Kernel Density Estimation
kde = sns.kdeplot(
    ax=ax,
    x=problematic_points_gdf['geometry'].x,
    y= problematic_points_gdf['geometry'].y,
    levels = levels,
    shade=True,
    cmap='Reds',
    alpha=0.9
)

# Transform to a vector
level_polygons = []
i=0
for col in kde.collections:
    # Loop through all polygons that have the same intensity level
    for contour in col.get_paths(): 
        paths = []
        # Create a polygon for the countour
        # First polygon is the main countour, the rest are holes
        for ncp,cp in enumerate(contour.to_polygons()):
            x = cp[:,0]
            y = cp[:,1]
            new_shape = Polygon([(i[0], i[1]) for i in zip(x,y)])
            if ncp == 0:
                poly = new_shape
            else:
                # Remove holes, if any
                poly = poly.difference(new_shape)

            # Append polygon to list
            paths.append(new_shape)
        paths.append(poly)
        # Create a MultiPolygon for the contour
        multi = MultiPolygon(paths)
        # Append MultiPolygon and level as tuple to list
        level_polygons.append((levels[i], multi))
        i+=1

intensity_gdf = gpd.GeoDataFrame(
    pd.DataFrame(level_polygons, columns =['level', 'geometry']), 
    geometry='geometry', crs = gdf.crs
)
# Save to file
filepath = os.path.join(OUTPUT_DIR, 'kde_polygons.gpkg')
intensity_gdf.to_file(filepath, driver='GPKG')
written_files.append(filepath)

print()
logger.info("The following files were written. Let's check them out!")
for written_file in written_files:
    logger.info(written_file)

print()

toc = time()
logger.success(f"Nothing left to be done: exiting. Elapsed time: {(toc-tic):.2f} seconds")