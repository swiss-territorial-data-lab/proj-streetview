INPUT_FOLDER=**

echo "Converting COCO datasets to YOLO datasets"
python scripts/utils/coco_to_yolo.py config/config_yolo.yaml
python scripts/utils/redistribute_images.py config/config_yolo.yaml
echo "Remove clipped images from conversion folder"
rm outputs/coco_for_yolo/$INPUT_FOLDER/*.jpg
echo "Done! Exiting..."