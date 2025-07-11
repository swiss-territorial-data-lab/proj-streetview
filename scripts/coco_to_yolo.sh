
echo "Converting COCO datasets to YOLO datasets"
python scripts/utils/coco_to_yolo.py config/config_yolo.yaml
python scripts/utils/redistribute_images.py config/config_yolo.yaml
echo "Erase COCO datasets"
rm -r outputs/COCO_datasets
echo "Done! Exiting..."