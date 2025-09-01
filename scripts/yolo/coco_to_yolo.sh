INPUT_FOLDER=coco_for_yolo_conversion_both

echo "Converting COCO datasets to YOLO datasets"
python scripts/utils/coco_to_yolo.py config/config_yolo.yaml
python scripts/utils/redistribute_images.py config/config_yolo.yaml
rm outputs/$INPUT_FOLDER/*.jpg
echo "Done! Exiting..."