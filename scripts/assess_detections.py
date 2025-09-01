#!/bin/python
# -*- coding: utf-8 -*-

import os
import sys
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yaml
from argparse import ArgumentParser
from time import time

from geopandas import GeoDataFrame
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from utils import misc
from utils import metrics
from utils.constants import CATEGORIES, COCO_FOR_YOLO_FOLDER, DONE_MSG, MODEL_FOLDER, SCATTER_PLOT_MODE

from loguru import logger
logger = misc.format_logger(logger)


def main(cfg_file_path):

    tic = time()
    logger.info('Starting...')

    logger.info(f"Using {cfg_file_path} as config file.")

    with open(cfg_file_path) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)[os.path.basename(__file__)]

    WORKING_DIR = cfg['working_directory']
    OUTPUT_DIR = cfg['output_folder']
    
    DATASETS = cfg['datasets']
    PATH_DETECTIONS = DATASETS['path_detections'].replace("<MODEL_FOLDER>", MODEL_FOLDER)
    DETECTION_FILES = DATASETS['detections_files']
    PATH_GROUND_TRUTH = DATASETS['path_ground_truth'].replace("<COCO_FOR_YOLO_FOLDER>", COCO_FOR_YOLO_FOLDER)
    GT_FILES = DATASETS['ground_truth_files']
    
    CONFIDENCE_THRESHOLD = cfg['confidence_threshold'] if 'confidence_threshold' in cfg.keys() else None
    IOU_THRESHOLD = cfg['iou_threshold'] if 'iou_threshold' in cfg.keys() else 0.25
    METHOD = cfg['metrics_method']
    DEBUG = False

    os.chdir(WORKING_DIR)
    logger.info(f'Working directory set to {WORKING_DIR}.')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    written_files = []
    
    # ------ Loading datasets

    # Class ids: monoclass case
    id_classes = [0]
    categories_info_df = pd.DataFrame(CATEGORIES[0], index=[0]).rename(columns={'id': 'label_class', 'name': 'category'})

    logger.info("Loading ground truth...")

    tiles_df_dict = {}
    labels_gdf_dict = {}
    label_segm_df_dict = {}
    nbr_tiles = 0
    nbr_labels = 0
    for dataset, labels_file in GT_FILES.items():
        with open(os.path.join(PATH_GROUND_TRUTH, labels_file)) as fp:
            coco_dict = json.load(fp)
        labels_df = pd.DataFrame.from_records(coco_dict['annotations'])

        # Get annotation geometry
        labels_df['geometry'] = labels_df['segmentation'].apply(lambda x: misc.segmentation_to_polygon(x))
        no_geom_tmp = labels_df[labels_df.geometry.isna()]
        if no_geom_tmp.shape[0] > 0:
            logger.warning(f"{no_geom_tmp.shape[0]} labels have no geometry in the {dataset} dataset with a max score of {round(no_geom_tmp['score'].max(), 2)}.")

        # Get annotation class
        labels_df.rename(columns={'category_id': 'label_class', 'id': 'label_id'}, inplace=True)
        labels_df = labels_df.merge(categories_info_df, on='label_class', how='left')
        label_segm_df_dict[dataset] = labels_df[['label_id', 'segmentation', 'bbox']].rename(columns={
            'segmentation': 'segmentation_labels', 'bbox': 'bbox_labels'
        })
        labels_df.drop(columns=['segmentation', 'iscrowd', 'supercategory', 'bbox'], inplace=True, errors='ignore')

        labels_gdf_dict[dataset] = GeoDataFrame(labels_df)

        # Format tile info
        all_aoi_tiles_df = pd.DataFrame.from_records(coco_dict['images']).rename(columns={'id': 'image_id'})
        all_aoi_tiles_df['file_name'] = [os.path.basename(path) for path in all_aoi_tiles_df['file_name']]
        tiles_df_dict[dataset] = all_aoi_tiles_df.copy()

        nbr_tiles += len(all_aoi_tiles_df)
        nbr_labels += len(labels_df)

    logger.success(f"{DONE_MSG} {nbr_tiles} tiles were found.")
    logger.success(f"{nbr_labels} labels were found.")

    # ------ Loading detections

    logger.info("Loading detections...")

    dets_gdf_dict = {}
    det_segm_df_dict = {}
    nbr_dets = 0
    for dataset, dets_file in DETECTION_FILES.items():
        with open(os.path.join(PATH_DETECTIONS, dets_file)) as fp:
            dets_dict = json.load(fp)
        if isinstance(dets_dict, dict):
            dets_dict = dets_dict['annotations']
        dets_df = pd.DataFrame.from_records(dets_dict)

        # Format detection info
        if 'image_id' not in dets_df.columns:
            dets_df = pd.merge(dets_df, tiles_df_dict[dataset][['file_name', 'image_id']], how='left', on='file_name')
        if 'det_id' not in dets_df.columns:
            dets_df.rename(columns={'id': 'det_id', 'category_id': 'det_class'}, inplace=True)
        dets_df['geometry'] = dets_df['segmentation'].apply(lambda x: misc.segmentation_to_polygon(x))
        no_geom_condition = dets_df.geometry.isna()
        if no_geom_condition.sum() > 0:
            logger.warning(f"{no_geom_tmp.shape[0]} detections have no geometry in the {dataset} dataset with a max score of {round(no_geom_tmp['score'].max(), 2)}.")
            dets_df = dets_df[~no_geom_condition]
        if DEBUG:
            logger.warning(f"Debug mode is on. Only 1/3 of the detections are kept.")
            dets_df = dets_df.sample(frac=0.33, random_state=42)

        det_segm_df_dict[dataset] = dets_df[['det_id', 'segmentation', 'bbox']].rename(columns={'segmentation': 'segmentation_dets', 'bbox': 'bbox_dets'})
        dets_df.drop(columns=['segmentation', 'bbox'], inplace=True)

        dets_gdf_dict[dataset] = GeoDataFrame(dets_df)
        logger.info(f"{len(dets_gdf_dict[dataset])} detections were found in the {dataset} dataset.")
        nbr_dets += len(dets_gdf_dict[dataset])

    logger.success(f"{DONE_MSG} {nbr_dets} detections were found.")

    del labels_df, all_aoi_tiles_df, dets_df, tiles_df_dict

    # initiate variables
    metrics_dict = {dataset: [] for dataset in dets_gdf_dict.keys()}
    metrics_dict_by_cl = {dataset: [] for dataset in dets_gdf_dict.keys()}
    metrics_df_dict = {}
    metrics_cl_df_dict = {}
    thresholds = np.arange(round(dets_gdf_dict['val'].score.min()*2, 1)/2, 1., 0.05)
   
    # ------ Comparing detections with ground-truth data and computing metrics

    # get metrics
    datasets_list = ["val"]
    outer_tqdm_log = tqdm(total=len(datasets_list), position=0)

    for dataset in datasets_list:

        outer_tqdm_log.set_description_str(f'Current dataset: {dataset}')
        inner_tqdm_log = tqdm(total=len(thresholds), position=1, leave=False)

        for threshold in thresholds:

            inner_tqdm_log.set_description_str(f'Threshold = {threshold:.2f}')

            tmp_dets_gdf = dets_gdf_dict[dataset][dets_gdf_dict[dataset].score >= threshold].copy()

            tagged_df_dict = metrics.get_fractional_sets(
                tmp_dets_gdf, 
                labels_gdf_dict[dataset],
                dataset,
                IOU_THRESHOLD
            )
            tp_k, fp_k, fn_k, p_k, r_k, precision, recall, f1 = metrics.get_metrics(id_classes=id_classes, method=METHOD, **tagged_df_dict)

            metrics_dict[dataset].append({
                'threshold': threshold, 
                'precision': precision, 
                'recall': recall, 
                'f1': f1
            })

            # label classes starting at 1 and detection classes starting at 0.
            for id_cl in id_classes:
                metrics_dict_by_cl[dataset].append({
                    'threshold': threshold,
                    'class': id_cl,
                    'precision_k': p_k[id_cl],
                    'recall_k': r_k[id_cl],
                    'TP_k' : tp_k[id_cl],
                    'FP_k' : fp_k[id_cl],
                    'FN_k' : fn_k[id_cl],
                })

            metrics_cl_df_dict[dataset] = pd.DataFrame.from_records(metrics_dict_by_cl[dataset])

            inner_tqdm_log.update(1)

        metrics_df_dict[dataset] = pd.DataFrame.from_records(metrics_dict[dataset])
        outer_tqdm_log.update(1)

    inner_tqdm_log.close()
    outer_tqdm_log.close()

    # let's generate some plots!

    fig = go.Figure()

    for dataset in datasets_list:
        # Plot of the precision vs recall

        fig.add_trace(
            go.Scatter(
                x=metrics_df_dict[dataset]['recall'],
                y=metrics_df_dict[dataset]['precision'],
                mode=SCATTER_PLOT_MODE,
                text=metrics_df_dict[dataset]['threshold'], 
                name=dataset
            )
        )

    fig.update_layout(
        xaxis_title="Recall",
        yaxis_title="Precision",
        xaxis=dict(range=[0., 1]),
        yaxis=dict(range=[0., 1])
    )

    file_to_write = os.path.join(OUTPUT_DIR, 'precision_vs_recall.html')
    fig.write_html(file_to_write)
    written_files.append(file_to_write)

    for dataset in datasets_list:
        # Generate a plot of TP, FN and FP for each class

        fig = go.Figure()

        for id_cl in id_classes:
            
            for y in ['TP_k', 'FN_k', 'FP_k']:

                fig.add_trace(
                    go.Scatter(
                            x=metrics_cl_df_dict[dataset]['threshold'][metrics_cl_df_dict[dataset]['class']==id_cl],
                            y=metrics_cl_df_dict[dataset][y][metrics_cl_df_dict[dataset]['class']==id_cl],
                            mode=SCATTER_PLOT_MODE,
                            name=y[0:2]+'_'+str(id_cl)
                        )
                    )

            fig.update_layout(xaxis_title="threshold", yaxis_title="#")
            
        if len(id_classes) > 1:
            file_to_write = os.path.join(OUTPUT_DIR, f'{dataset}_TP-FN-FP_vs_threshold_dep_on_class.html')

        else:
            file_to_write = os.path.join(OUTPUT_DIR, f'{dataset}_TP-FN-FP_vs_threshold.html')

        fig.write_html(file_to_write)
        written_files.append(file_to_write)

        fig = go.Figure()

        for y in ['precision', 'recall', 'f1']:

            fig.add_trace(
                go.Scatter(
                    x=metrics_df_dict[dataset]['threshold'],
                    y=metrics_df_dict[dataset][y],
                    mode=SCATTER_PLOT_MODE,
                    name=y
                )
            )

        fig.update_layout(xaxis_title="threshold")

        file_to_write = os.path.join(OUTPUT_DIR, f'{dataset}_metrics_vs_threshold.html')
        fig.write_html(file_to_write)
        written_files.append(file_to_write)


    # ------ tagging detections

    # we select the threshold which maximizes the f1-score on the val dataset or the one passed by the user
    if 'val' in metrics_df_dict.keys() and CONFIDENCE_THRESHOLD:
        logger.error('The confidence threshold was determined over the val dataset, but a confidence threshold is given in the config file.')
        logger.error(f'confidence threshold: val dataset = {metrics_df_dict["val"].loc[metrics_df_dict["val"]["f1"].argmax(), "threshold"]}, config = {CONFIDENCE_THRESHOLD}')
        logger.warning('The confidence threshold from the config file is used.')
    if CONFIDENCE_THRESHOLD:
        selected_threshold = CONFIDENCE_THRESHOLD
        logger.info(f"Tagging detections with threshold = {selected_threshold:.2f}, which is the threshold given in the config file.")
    elif 'val' in metrics_df_dict.keys():
        selected_threshold = metrics_df_dict['val'].loc[metrics_df_dict['val']['f1'].argmax(), 'threshold']
        logger.info(f"Tagging detections with threshold = {selected_threshold:.2f}, which maximizes the f1-score on the val dataset.")
    else:
        raise AttributeError('No confidence threshold can be determined without the validation dataset or the passed value.')

    tagged_dets_gdf_dict = {}

    # TRUE/FALSE POSITIVES, FALSE NEGATIVES

    logger.info(f'Method to compute the metrics = {METHOD}')

    global_metrics_dict = {'dataset': [], 'precision': [], 'recall': [], 'f1': []}
    metrics_cl_df_dict = {}     # re-initialisation of the variable
    metrics_dict_by_cl = {dataset: [] for dataset in dets_gdf_dict.keys()}     # re-initialisation of the variable
    for dataset in metrics_dict.keys():

        tmp_dets_gdf = dets_gdf_dict[dataset][dets_gdf_dict[dataset].score >= selected_threshold].copy()
        logger.info(f'Number of detections = {len(tmp_dets_gdf)}')
        logger.info(f'Number of labels = {len(labels_gdf_dict[dataset])}')

        tagged_df_dict = metrics.get_fractional_sets(
            tmp_dets_gdf, 
            labels_gdf_dict[dataset],
            dataset,
            IOU_THRESHOLD
        )

        tp_k, fp_k, fn_k, p_k, r_k, precision, recall, f1 = metrics.get_metrics(id_classes=id_classes, method=METHOD, **tagged_df_dict)
        global_metrics_dict['dataset'].append(dataset)
        global_metrics_dict['precision'].append(precision)
        global_metrics_dict['recall'].append(recall)
        global_metrics_dict['f1'].append(f1)
        logger.info(f'Dataset = {dataset} => precision = {precision:.3f}, recall = {recall:.3f}, f1 = {f1:.3f}')

        # label classes starting at 1 and detection classes starting at 0.
        for id_cl in id_classes:
            metrics_dict_by_cl[dataset].append({
                'threshold': selected_threshold,
                'class': id_cl,
                'precision_k': p_k[id_cl],
                'recall_k': r_k[id_cl],
                'TP_k' : tp_k[id_cl],
                'FP_k' : fp_k[id_cl],
                'FN_k' : fn_k[id_cl],
            })

        metrics_cl_df_dict[dataset] = pd.DataFrame.from_records(metrics_dict_by_cl[dataset])

        tagged_df_dict["fn_df"] = tagged_df_dict["fn_df"].merge(label_segm_df_dict[dataset], how='left', on='label_id').rename(
            columns={'segmentation_labels': 'segmentation', 'bbox_labels': 'bbox'}
        )
        for key in ['tp_df', 'fp_df']:
            tagged_df_dict[key] = tagged_df_dict[key].merge(det_segm_df_dict[dataset], how='left', on='det_id').rename(
                columns={'segmentation_dets': 'segmentation', 'bbox_dets': 'bbox'}
            )
        tagged_dets_gdf_dict[dataset] = pd.concat(tagged_df_dict.values())

    tagged_dets_df = pd.concat([tagged_dets_gdf_dict[x] for x in metrics_dict.keys()])
    tagged_dets_df['det_category'] = [
        categories_info_df.loc[categories_info_df.label_class==det_class+1, 'category'].iloc[0] 
        if not np.isnan(det_class) else None
        for det_class in tagged_dets_df.det_class
    ]

    cols = [
        'dataset', 'tag', 
        'label_id', 'label_class', 'category', 
        'det_id', 'score', 'det_class', 'det_category', 'area', 'IOU', 'segmentation', 'bbox', 
        'image_id'
    ]
    file_to_write = os.path.join(OUTPUT_DIR, 'tagged_detections.json')
    # Get the segmentation and bbox back
    with open(file_to_write, 'w') as fp:
        tagged_dets_df[cols].to_json(fp, orient='records')
    written_files.append(file_to_write)

    # Save the metrics by class for each dataset
    metrics_by_cl_df = pd.DataFrame()
    for dataset in metrics_cl_df_dict.keys():
        dataset_df = metrics_cl_df_dict[dataset].copy()
        dataset_df['dataset'] = dataset
        dataset_df.drop(columns=['threshold'], inplace=True)

        metrics_by_cl_df = pd.concat([metrics_by_cl_df, dataset_df], ignore_index=True)
    
    metrics_by_cl_df['category'] = [
        categories_info_df.loc[categories_info_df.label_class==det_class+1, 'category'].iloc[0] 
        for det_class in metrics_by_cl_df['class'].to_numpy()
    ] 

    file_to_write = os.path.join(OUTPUT_DIR, 'metrics_by_class.csv')
    metrics_by_cl_df[
        ['class', 'category', 'TP_k', 'FP_k', 'FN_k', 'precision_k', 'recall_k', 'dataset']
    ].sort_values(by=['dataset', 'class']).to_csv(file_to_write, index=False)
    written_files.append(file_to_write)

    tmp_df = metrics_by_cl_df[['dataset', 'TP_k', 'FP_k', 'FN_k']].groupby(by='dataset', as_index=False).sum()
    tmp_df2 = pd.DataFrame(global_metrics_dict, index = range(len(dets_gdf_dict.keys())))
    global_metrics_df = tmp_df.merge(tmp_df2, on='dataset')
    global_metrics_df.rename({'TP_k': 'TP', 'FP_k': 'FP', 'FN_k': 'FN', 'precision_k': 'precision', 'recall_k': 'recall'}, inplace=True)

    file_to_write = os.path.join(OUTPUT_DIR, 'global_metrics.csv')
    global_metrics_df.to_csv(file_to_write, index=False)
    written_files.append(file_to_write)

    # Save the confusion matrix
    na_value_category = tagged_dets_df.category.isna()
    sorted_classes = tagged_dets_df.loc[~na_value_category, 'category'].sort_values().unique().tolist() + ['background']
    tagged_dets_df.loc[na_value_category, 'category'] = 'background'
    tagged_dets_df.loc[tagged_dets_df.det_category.isna(), 'det_category'] = 'background'
    
    for dataset in tagged_dets_df.dataset.unique():
        tagged_dataset_df = tagged_dets_df[tagged_dets_df.dataset == dataset].copy()

        true_class = tagged_dataset_df.category.to_numpy()
        detected_class = tagged_dataset_df.det_category.to_numpy()

        confusion_array = confusion_matrix(true_class, detected_class, labels=sorted_classes)
        confusion_df = pd.DataFrame(confusion_array, index=sorted_classes, columns=sorted_classes, dtype='int64')
        confusion_df.rename(columns={'background': 'missed labels'}, inplace=True)

        file_to_write = os.path.join(OUTPUT_DIR, f'{dataset}_confusion_matrix.csv')
        confusion_df.to_csv(file_to_write)
        written_files.append(file_to_write)


    # ------ wrap-up

    print()
    logger.info("The following files were written. Let's check them out!")
    for written_file in written_files:
        logger.info(written_file)

    print()

    toc = time()
    logger.success(f"Nothing left to be done: exiting. Elapsed time: {(toc-tic):.2f} seconds")

    sys.stderr.flush()


if __name__ == "__main__":

    parser = ArgumentParser(description="This script assesses the quality of detections with respect to ground-truth/other labels.")
    parser.add_argument('config_file', type=str, help='a YAML config file')
    args = parser.parse_args()

    main(args.config_file)