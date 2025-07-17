#  Imports des bibliothèques nécessaires
import os
from argparse import ArgumentParser
from loguru import logger
from yaml import FullLoader, load

import torch
import ray
from ray import tune                                       # Ray Tune pour optimisation d’hyperparamètres
from ultralytics import YOLO

from utils.constants import TILE_SIZE
from utils.misc import format_logger

logger = format_logger(logger)

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

    # 易 Sélection du bon GPU (si disponible)
    local_rank = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    #  Lance l’entraînement avec les hyperparamètres fournis
    _ = model.train(
        data=YOLO_FILE,           # Fichier YAML du dataset
        epochs=25,                     # Nombre d’époques
        lr0=config["lr0"],             # Taux d’apprentissage initial
        batch=config["batch"],         # Taille de batch
        optimizer=config["optimizer"], # Optimiseur (SGD, Adam, etc.)
        patience=config["patience"],
        imgsz=TILE_SIZE,
        device=device,
        workers=0                      # Pas de workers multi-thread pour éviter bugs Ray
    )

logger.info('Starting...')

parser = ArgumentParser(description="This script redistributes images between datasets.")
parser.add_argument('config_file', type=str, help='a YAML config file')
args = parser.parse_args()

logger.info(f"Using {args.config_file} as config file.")

cfg_file_path = args.config_file

with open(cfg_file_path) as fp:
    cfg = load(fp, Loader=FullLoader)[os.path.basename(__file__)]

WORKING_DIR = cfg["working_directory"]
OUTPUT_DIR = cfg["output_folder"]
YOLO_FILE = cfg["yolo_file"]

os.chdir(WORKING_DIR)  # Répertoire du projet YOLO

#  Affiche le nombre de GPU disponibles
print(f"Available GPUs: {torch.cuda.device_count()}")

#  Initialise proprement Ray
ray.shutdown()  # Arrête tout cluster Ray existant
ray.init()      # Démarre un nouveau cluster Ray local

# ️ Affiche les ressources détectées par Ray
print(ray.cluster_resources())

# 離 Espace de recherche (grille simple ici, peut être étendu)
search_space = {
    "lr0": tune.grid_search([0.005, 0.1, 0.005]),  # Taux d’apprentissage initial
    "batch": tune.grid_search([5, 20]),            # Taille de batch
    "optimizer": tune.grid_search(["SGD", "Adam"]), # Optimiseur
    "model": tune.grid_search([
        "yolo11n-seg", "yolo11s-seg", "yolov8n-seg", "yolov8s-seg"
    ]),  # Modèles légers/rapides
    'patience': tune.grid_search([25, 100])
}

# ⚙️ Lance les expériences Ray Tune
analysis = tune.run(
    train_yolo,                                  # Fonction à appeler
    config=search_space,                         # Espace de recherche
    resources_per_trial={"cpu": 1, "gpu": 0.1},  # Ressources par expérience (fraction GPU)
    num_samples=1,                                # Nombre de tirages (utile avec random/grid search)
    local_dir=OUTPUT_DIR,                        # Dossier de sortie
)
