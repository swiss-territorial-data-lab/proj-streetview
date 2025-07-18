import os
from tqdm import tqdm

from pandas import DataFrame

def yolo_to_coco_annotations(results, image_info_df=None, tqdm_disable=False):
    if isinstance(image_info_df, DataFrame):
        image_info_as_df = True
        _image_info_df = image_info_df.copy()
        _image_info_df['file_name'] = _image_info_df['file_name'].apply(lambda x: os.path.basename(x))
    else:
        image_info_as_df = False
        image_id = None

    det_id = 0
    annotations = []

    for result in tqdm(results, desc="Converting annotations", disable=tqdm_disable):
        if image_info_as_df:
            image_info = _image_info_df[_image_info_df['file_name'] == os.path.basename(result.path)]
            image_id = image_info['image_id'].iloc[0]

        for det_index in range(len(result.boxes.cls)):
            category_id = int(result.boxes.cls[det_index])

            # Get segment
            transformed_coords = [int(coord) for coord in result.masks.xy[det_index].flatten().tolist()]
            
            # Get bbox
            bbox = [int(coord) for coord in result.boxes.xywh[det_index].tolist()]

            # Get score
            score = round(result.boxes.conf[det_index].tolist(), 3)

            # Get and control area
            area = int(result.masks.data[det_index].sum().tolist())
            if area == 0:
                print(f"Found an empty mask with score {score}...")
                continue

            # Create annotation
            annotation = {
                "id": det_id,
                "image_id": image_id,
                "bbox": bbox,
                "area": area,
                "score": score,
                "category_id": category_id,
                "segmentation": transformed_coords,
            }
            annotations.append(annotation)

            det_id += 1

    return annotations