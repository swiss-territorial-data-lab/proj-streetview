import pandas as pd
from geopandas import sjoin

def get_fractional_sets(dets_df, labels_df, dataset, iou_threshold=0.25):
    """
    Find the intersecting detections and labels.
    Control their IoU and class to get the TP.
    Tag detections and labels not intersecting or not intersecting enough as FP and FN respectively.
    Save the intersections with mismatched class ids in a separate geodataframe.

    Args:
        dets_df (geodataframe): geodataframe of the detections.
        labels_df (geodataframe): geodataframe of the labels.
        iou_threshold (float): threshold to apply on the IoU to determine if detections and labels can be matched. Defaults to 0.25.

    Returns:
        dict:
        - geodataframe: true positive intersections between a detection and a label;
        - geodataframe: false postive detections;
        - geodataframe: false negative labels;
        """

    _dets_gdf = dets_df.reset_index(drop=True)
    _labels_gdf = labels_df.reset_index(drop=True)

    columns_list = ['area', 'geometry', 'dataset', 'label_id', 'label_class']
    fp_df = pd.DataFrame(columns=_dets_gdf.columns)
    tp_df = pd.DataFrame(columns=columns_list + ['det_id', 'det_class', 'score', 'IOU'])
    fn_df = pd.DataFrame(columns=columns_list)
    tagged_df_dict = {'tp_df': tp_df, 'fp_df': fp_df, 'fn_df': fn_df}
       
    if len(_labels_gdf) == 0:
        tagged_df_dict['FP'] = _dets_gdf
        return tagged_df_dict
    
    # we add a id column to the labels dataset, which should not exist in detections too;
    # this allows us to distinguish matching from non-matching detections
    _labels_gdf.rename(columns={'id': 'label_id'}, inplace=True)
    _dets_gdf['det_id'] = _dets_gdf.index
    # We need to keep both geometries after sjoin to check the best intersection over union
    _labels_gdf['label_geom'] = _labels_gdf.geometry

    # TRUE POSITIVES
    left_join = pd.DataFrame()
    if dataset == 'trn':
        for image in _dets_gdf.image_id.unique():
            left_join = pd.concat([
                left_join, sjoin(
                    _dets_gdf[_dets_gdf.image_id == image], 
                    _labels_gdf[_labels_gdf.image_id == image], 
                    how='left', predicate='intersects', lsuffix='dets', rsuffix='labels'
                )
            ], ignore_index=True)
        # Test that something is detected
        candidates_tp_gdf = left_join[left_join.label_id.notnull()].copy()
    else:
        left_join = sjoin(_dets_gdf, _labels_gdf, how='left', predicate='intersects', lsuffix='dets', rsuffix='labels')
        candidates_tp_gdf = left_join[left_join.image_id_labels == left_join.image_id_dets]  # image_id_labels not null iff label_id not null

    # IoU computation between labels and detections
    geom1 = candidates_tp_gdf['geometry'].to_numpy().tolist()
    geom2 = candidates_tp_gdf['label_geom'].to_numpy().tolist()    
    candidates_tp_gdf.loc[:, ['IOU']] = [intersection_over_union(i, ii) for (i, ii) in zip(geom1, geom2)]

    # Filter detections based on IoU value
    best_matches_gdf = candidates_tp_gdf.groupby(['det_id'], group_keys=False).apply(lambda g:g[g.IOU==g.IOU.max()])
    best_matches_gdf.drop_duplicates(subset=['det_id'], inplace=True) # Case to IoU are equal for a detection

    # Detection, resp labels, with IOU lower than threshold value are considered as FP, resp FN, and saved as such
    actual_matches_gdf = best_matches_gdf[best_matches_gdf['IOU'] >= iou_threshold].copy()
    actual_matches_gdf = actual_matches_gdf.sort_values(by=['IOU'], ascending=False).drop_duplicates(subset=['label_id', 'image_id_labels'])
    actual_matches_gdf['IOU'] = actual_matches_gdf.IOU.round(3)

    del best_matches_gdf, candidates_tp_gdf

    # Test that labels and detections share the same class (id starting at 1 for labels and at 0 for detections)
    condition = actual_matches_gdf.label_class == actual_matches_gdf.det_class + 1
    tp_df = actual_matches_gdf[condition].reset_index(drop=True)
    assert len(tp_df) == len(actual_matches_gdf), "Unmatched class in the mono-class case"
    tp_df['tag'] = 'TP'
    tp_df = tp_df.drop(columns=['index_labels', 'image_id_labels', 'geometry', 'label_geom']).rename(columns={'image_id_dets': 'image_id'})
    tagged_df_dict['tp_df'] = pd.concat([tagged_df_dict['tp_df'], tp_df], ignore_index=True)

    matched_det_ids = tp_df['det_id'].unique().tolist()
    matched_label_ids = tp_df['label_id'].unique().tolist()

    del actual_matches_gdf, tp_df

    # FALSE POSITIVES
    fp_df = _dets_gdf[~_dets_gdf.det_id.isin(matched_det_ids)].copy()
    assert all(~fp_df.duplicated()), "Some detections were duplicated."
    fp_df.drop(columns='geometry', inplace=True)
    fp_df['tag'] = 'FP'
    tagged_df_dict['fp_df'] = pd.concat([tagged_df_dict['fp_df'], fp_df], ignore_index=True)

    del fp_df, left_join

    # FALSE NEGATIVES
    fn_df = _labels_gdf[~_labels_gdf.label_id.isin(matched_label_ids)].copy()
    assert all(~fn_df.duplicated()), "Some labels were duplicated."
    fn_df.drop(columns=['geometry', 'label_geom'], inplace=True)
    fn_df['tag'] = 'FN'
    fn_df['dataset'] = dataset
    tagged_df_dict['fn_df'] = pd.concat([tagged_df_dict['fn_df'], fn_df], ignore_index=True)

    assert len(tagged_df_dict['tp_df']) + len(tagged_df_dict['fn_df']) == len(_labels_gdf), "Some labels went missing or were duplicated."
    assert len(tagged_df_dict['tp_df']) + len(tagged_df_dict['fp_df']) == len(_dets_gdf), "Some detections went missing or were duplicated."

    return tagged_df_dict


def get_metrics(tp_df, fp_df, fn_df, id_classes=0, method='macro-average'):
    """Determine the metrics based on the TP, FP and FN

    Args:
        tp_df (geodataframe): true positive detections
        fp_df (geodataframe): false positive detections
        fn_df (geodataframe): false negative labels
        id_classes (list): list of the possible class ids. Defaults to 0.
        method (str): method used to compute multi-class metrics. Default to macro-average
    
    Returns:
        tuple: 
            - dict: TP count for each class
            - dict: FP count for each class
            - dict: FN count for each class
            - dict: precision for each class
            - dict: recall for each class
            - dict: f1-score for each class
            - float: accuracy
            - float: precision;
            - float: recall;
            - float: f1 score.
    """

    by_class_dict = {key: 0 for key in id_classes}
    tp_k = by_class_dict.copy()
    fp_k = by_class_dict.copy()
    fn_k = by_class_dict.copy()
    p_k = by_class_dict.copy()
    r_k = by_class_dict.copy()
    count_k = by_class_dict.copy()
    pw_k = by_class_dict.copy()
    rw_k = by_class_dict.copy()

    total_labels = len(tp_df) + len(fn_df)
    for id_cl in id_classes:

        fp_count = len(fp_df[fp_df.det_class==id_cl])
        fn_count = len(fn_df[fn_df.label_class==id_cl+1])  # label class starting at 1 and id class at 0
        tp_count = len(tp_df[tp_df.det_class==id_cl])

        fp_k[id_cl] = fp_count
        fn_k[id_cl] = fn_count
        tp_k[id_cl] = tp_count

        count_k[id_cl] = tp_count + fn_count
        if tp_count > 0:
            p_k[id_cl] = tp_count / (tp_count + fp_count)
            r_k[id_cl] = tp_count / (tp_count + fn_count)

        if (method == 'macro-weighted-average') & (total_labels > 0):
            pw_k[id_cl] = (count_k[id_cl] / total_labels) * p_k[id_cl]
            rw_k[id_cl] = (count_k[id_cl] / total_labels) * r_k[id_cl] 

    if method == 'macro-average':   
        precision = sum(p_k.values()) / len(id_classes)
        recall = sum(r_k.values()) / len(id_classes)
    elif method == 'macro-weighted-average':  
        precision = sum(pw_k.values()) / len(id_classes)
        recall = sum(rw_k.values()) / len(id_classes)
    elif method == 'micro-average':  
        if sum(tp_k.values()) == 0:
            precision = 0
            recall = 0
        else:
            precision = sum(tp_k.values()) / (sum(tp_k.values()) + sum(fp_k.values()))
            recall = sum(tp_k.values()) / (sum(tp_k.values()) + sum(fn_k.values()))

    if precision==0 and recall==0:
        return tp_k, fp_k, fn_k, p_k, r_k, 0, 0, 0
    
    f1 = 2 * precision * recall / (precision + recall)
    
    return tp_k, fp_k, fn_k, p_k, r_k, precision, recall, f1


def intersection_over_union(polygon1_shape, polygon2_shape):
    """Determine the intersection area over union area (IOU) of two polygons

    Args:
        polygon1_shape (geometry): first polygon
        polygon2_shape (geometry): second polygon

    Returns:
        int: Unrounded ratio between the intersection and union area
    """

    # Calculate intersection and union, and the IOU
    polygon_intersection = polygon1_shape.intersection(polygon2_shape).area
    polygon_union = polygon1_shape.area + polygon2_shape.area - polygon_intersection

    if polygon_union != 0:
        iou = polygon_intersection / polygon_union
    else:
        iou = 0

    return iou