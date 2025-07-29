DONE_MSG = "...done."
SCATTER_PLOT_MODE = 'markers+lines'
CATEGORIES = [{'id': 1, 'name': 'manhole', 'supercategory': 'round plate'}]
TILE_SIZE = 512
YOLO_TRAINING_PARAMS = {
    'data': '/mnt/data-volume-02/gsalamin/GitHub/proj-streetview/config/yolo/yolo_dataset.yaml',
    'imgsz': TILE_SIZE,
    'multi_scale': True,
    'translate': 0,
    # 'single_cls': True
}