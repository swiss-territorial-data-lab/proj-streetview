import os

def yolo_to_coco_annotations(result, image_info_df=None, image_info_as_df=False):

    if image_info_as_df:
        _image_info = image_info_df[image_info_df['file_name'] == os.path.basename(result.path)]
        image_id = int(_image_info['id'].iloc[0])
        image_file = _image_info['file_name'].iloc[0]

    annotations = []
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
            "image_id": image_id,
            "bbox": bbox,
            "area": area,
            "score": score,
            "det_class": category_id,
            "segmentation": [transformed_coords],
            "file_name": image_file
        }
        annotations.append(annotation)

    return annotations