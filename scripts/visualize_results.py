import cv2
import json
import os
from argparse import ArgumentParser
from loguru import logger
from time import time
from tqdm import tqdm
from yaml import load, FullLoader

from pandas import DataFrame

from utils.misc import format_logger

logger = format_logger(logger)

parser = ArgumentParser(description="This script prepares COCO datasets.")
parser.add_argument('config_file', type=str, help='a YAML config file')
args = parser.parse_args()

tic = time()
logger.info('Starting...')

logger.info(f"Using {args.config_file} as config file.")

with open(args.config_file) as fp:
    cfg = load(fp, Loader=FullLoader)[os.path.basename(__file__)]

WORKING_DIR = cfg['working_directory']
OUTPUT_DIR = cfg['output_folder']
IMAGE_DIR = cfg['image_folder']

TAGGED_COCO_FILES = cfg['tagged_COCO_files']

os.chdir(WORKING_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

logger.info("Registering COCO datasets...")
images = []
annotations = []
for dataset in TAGGED_COCO_FILES.keys():
    with open(TAGGED_COCO_FILES[dataset]) as fp:
        coco_data = json.load(fp)
        if isinstance(coco_data, dict):
            images.extend(coco_data['images'])
            annotations.extend(coco_data['annotations'])
        else:
            annotations.extend(coco_data)
            with open(cfg['coco_file_for_images']) as fp:
                images.extend(json.load(fp)['images'])

del coco_data

images_df = DataFrame.from_records(images).drop_duplicates(subset=["file_name"])
annotations_df = DataFrame.from_records(annotations)
if 'id' not in annotations_df.columns:
    annotations_df['id'] = [det_id if det_id == None else label_id for det_id, label_id in zip(annotations_df.det_id, annotations_df.label_id)]

logger.info("Let's tag some sample images...")
colors_dict = {
    "TP": (0, 255, 0),
    "FP": (247, 195, 79),
    "FN": (37, 168, 249)
}
# https://stackoverflow.com/questions/50805634/how-to-create-mask-images-from-coco-dataset
# https://opencv.org/blog/image-annotation-using-opencv/
nbr_images = 50
for coco_image in tqdm(images_df.sample(n=nbr_images, random_state=42).itertuples(), desc="Tagging images", total=nbr_images):
    output_filename = f'det_{coco_image.file_name.split("/")[-1]}'
    output_filename = output_filename.replace('tif', 'png')
    im = cv2.imread(coco_image.file_name if str(coco_image.file_name).startswith(IMAGE_DIR) else os.path.join(IMAGE_DIR, coco_image.file_name))
    corresponding_annotations = annotations_df[annotations_df["image_id"] == coco_image.id]
    if corresponding_annotations.empty:
        continue
    for ann in corresponding_annotations.itertuples():
        if 'tag' in corresponding_annotations.columns:
            color = colors_dict[ann.tag]
        else:
            color = (255, 0, 0)
        bbox = [int(b) for b in ann.bbox]
        cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), color, 2)

        text_position = {
            "trn": (bbox[0], bbox[1]-10),
            "val": (bbox[0], bbox[1] + 10),
            "tst": (bbox[0], bbox[1] + bbox[3] + 10),
        }   
        cv2.putText(im, ' '.join([ann.dataset, str(ann.id), str(round(ann.score, 2))] + ([ann.tag] if 'tag' in corresponding_annotations.columns else [])), text_position[ann.dataset], cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    filepath = os.path.join(OUTPUT_DIR, output_filename)
    cv2.imwrite(filepath, im)

logger.success(f"Done! {nbr_images} images were tagged and saved in {os.path.join(WORKING_DIR, OUTPUT_DIR)}.")
logger.info(f"Done in {round(time() - tic, 2)} seconds.")