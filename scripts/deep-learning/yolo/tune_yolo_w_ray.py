import os
import sys
from argparse import ArgumentParser
from loguru import logger
from yaml import FullLoader, load

import torch
import ray
from ray import tune
from ultralytics import YOLO

sys.path.insert(1, 'scripts')
from utils.constants import YOLO_TRAINING_PARAMS
from utils.misc import fill_path, format_logger

logger = format_logger(logger)

# TODO: improve script based on https://docs.ultralytics.com/integrations/ray-tune/#ray-tune

#  Fonction principale d’entraînement
def train_yolo(config):
    """
    Fonction appelée par Ray Tune.
    Elle entraîne un modèle YOLO avec les hyperparamètres passés en config,
    et enregistre les résultats sur wandb.
    """
    print(os.getcwd())

    #  Charge un modèle YOLO pré-entraîné (ex: yolov8n.pt)
    model = YOLO(config["model"] + ".pt")
    config.pop("model")

    # 易 Sélection du bon GPU (si disponible)
    local_rank = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    if not os.path.exists(YOLO_TRAINING_PARAMS['data']):
        YOLO_TRAINING_PARAMS['data'] = os.path.join('/app', 'config', YOLO_TRAINING_PARAMS['data'].split('config/')[-1])

    #  Lance l’entraînement avec les hyperparamètres fournis
    _ = model.train(
        device=device,
        workers=0,                      # Pas de workers multi-thread pour éviter bugs Ray
        **config,
        **YOLO_TRAINING_PARAMS
    )

logger.info('Starting...')

parser = ArgumentParser(description="This script redistributes images between datasets.")
parser.add_argument('config_file', type=str, help='a YAML config file')
args = parser.parse_args()

logger.info(f"Using {args.config_file} as config file.")

cfg_file_path = args.config_file

with open(cfg_file_path) as fp:
    cfg = load(fp, Loader=FullLoader)[os.path.basename(__file__)]

OUTPUT_DIR = fill_path([cfg["output_folder"]])[0]

os.makedirs(OUTPUT_DIR, exist_ok=True)

#  Affiche le nombre de GPU disponibles
print(f"Available GPUs: {torch.cuda.device_count()}")

#  Initialise proprement Ray
ray.shutdown()  # Arrête tout cluster Ray existant
ray.init()      # Démarre un nouveau cluster Ray local

# ️ Affiche les ressources détectées par Ray
print(ray.cluster_resources())

# 離 Espace de recherche (grille simple ici, peut être étendu)
search_space = {
    "model": tune.grid_search(["yolo11s-seg", "yolo11m-seg"]),
    "lr0": tune.grid_search([0.003, 0.005, 0.01]),
    "optimizer": tune.grid_search(["SGD", "Adam"]),
    "epochs": tune.grid_search([50, 75]),
}

# ⚙️ Lance les expériences Ray Tune
analysis = tune.run(
    train_yolo,                                  # Fonction à appeler
    config=search_space,                         # Espace de recherche
    resources_per_trial={"cpu": 1, "gpu": 1},  # Ressources par expérience (fraction GPU)
    num_samples=1,                                # Nombre de tirages (utile avec random/grid search)
    storage_path=OUTPUT_DIR,                        # Dossier de sortie
)
