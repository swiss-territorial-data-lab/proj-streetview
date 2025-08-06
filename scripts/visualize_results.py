import cv2
import json
import os
from argparse import ArgumentParser
from loguru import logger
from time import time
from tqdm import tqdm
from yaml import load, FullLoader

from pandas import DataFrame

from utils.constants import IMAGE_DIR
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

TAGGED_COCO_FILES = cfg['tagged_COCO_files']

IMAGE_IDS = cfg['image_ids'] if 'image_ids' in cfg.keys() else []

os.chdir(WORKING_DIR)
if os.path.exists(OUTPUT_DIR):
    os.system(f"rm -rf {OUTPUT_DIR}")
os.makedirs(OUTPUT_DIR)

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
        for im_dataset in cfg['coco_file_for_images'].keys():
            with open(cfg['coco_file_for_images'][im_dataset]) as fp:
                images.extend(json.load(fp)['images'])
    test=1

del coco_data

images_df = DataFrame.from_records(images).drop_duplicates(subset=["file_name"])
if len (IMAGE_IDS) > 0:
    sample_images_df = images_df[images_df['id'].isin(IMAGE_IDS)]
else:
    sample_images_df = images_df.sample(frac=1, random_state=42)    # sample = shuffle rows to create mixity in output
annotations_df = DataFrame.from_records(annotations)
if 'id' not in annotations_df.columns:
    annotations_df['id'] = [det_id if det_id == None else label_id for det_id, label_id in zip(annotations_df.det_id, annotations_df.label_id)]

logger.info("Let's tag some sample images...")
if 'tag' in annotations_df.columns and any(annotations_df.tag.isna()):
    score_threshold = annotations_df.loc[annotations_df.tag.notna(), 'score'].min()
    logger.info(f'A threshold of {score_threshold} is applied on the score.')
    annotations_df.loc[annotations_df.tag.isna(), 'tag'] = 'oth'
    annotations_df = annotations_df[annotations_df.score >= score_threshold]

colors_dict = {
    "TP": (0, 255, 0),
    "FP": (247, 195, 79),
    "FN": (37, 168, 249),
    "oth": (0, 0, 0)
}
nbr_images_per_dataset = 50
images_pro_dataset = {key: 0 for key in annotations_df["dataset"].unique()}
nbr_images = nbr_images_per_dataset*len(images_pro_dataset.keys())
for coco_image in tqdm(sample_images_df.itertuples(), desc="Tagging images"):
    if all([im_nbr >= 50 for im_nbr in images_pro_dataset.values()]):
        break

    corresponding_annotations = annotations_df[annotations_df["image_id"] == coco_image.id].reset_index(drop=True)
    if corresponding_annotations.empty:
        continue
    dataset = corresponding_annotations.loc[0, 'dataset']
    if images_pro_dataset[dataset] >= 50:
        continue
    images_pro_dataset[dataset] += 1

    image_dir = IMAGE_DIR[coco_image.dataset]

    output_filename = f'{dataset}_det_{coco_image.file_name.split("/")[-1]}'.replace('tif', 'png')
    input_path = os.path.join(image_dir, os.path.basename(coco_image.file_name))
    im = cv2.imread(input_path)
    if im is None:
        logger.warning(f"Image {input_path} not found.")
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
            "val": (bbox[0], bbox[1] + bbox[3] + 20),
            "tst": (bbox[0], bbox[1] + bbox[3] + 20),
            "oth": (bbox[0], bbox[1] + bbox[3] + 20)
        }
        cv2.putText(im, ' '.join([ann.dataset, str(ann.id), str(round(ann.score, 2))] + ([ann.tag] if 'tag' in corresponding_annotations.columns else [])), text_position[ann.dataset], cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    filepath = os.path.join(OUTPUT_DIR, output_filename)
    cv2.imwrite(filepath, im)

logger.success(f"Done! {nbr_images} images were tagged and saved in {os.path.join(WORKING_DIR, OUTPUT_DIR)}.")
logger.info(f"Done in {round(time() - tic, 2)} seconds.")