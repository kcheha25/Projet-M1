import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
import numpy as np
import os, json, cv2, random
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from matplotlib import pyplot as plt
import torch
from detectron2.utils.logger import setup_logger
from detectron2.data import build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data.datasets import register_coco_instances
from PIL import Image
import distutils.core
from detectron2.engine import DefaultTrainer

dist = distutils.core.run_setup("setup.py")

# Enregistrement des instances COCO pour les ensembles d'entraînement, de validation et de test
register_coco_instances("my_dataset_train", {}, "projetm1_data/train/_annotations.coco.json", "projetm1_data/train")
register_coco_instances("my_dataset_val", {}, "projetm1_data/valid/_annotations.coco.json", "projetm1_data/valid")
register_coco_instances("my_dataset_test", {}, "projetm1_data/test/_annotations.coco.json", "projetm1_data/test")

# Obtention des métadonnées et des ensembles de données pour les ensembles, d'entraînement et de validation 
test_metadata = MetadataCatalog.get("my_dataset_test")
test_dataset_dicts = DatasetCatalog.get("my_dataset_test")
train_metadata = MetadataCatalog.get("my_dataset_train")
train_dataset_dicts = DatasetCatalog.get("my_dataset_train")
val_metadata = MetadataCatalog.get("my_dataset_val")
val_dataset_dicts = DatasetCatalog.get("my_dataset_val")

# Utilisation d'une boucle pour afficher deux images aléatoires de l'ensemble d'entraînement avec leurs annotations
for d in random.sample(train_dataset_dicts, 2):
  img = cv2.imread(d["file_name"])
  visualizer = Visualizer(img[:, :, ::-1], metadata=train_metadata, scale=0.5)
  vis = visualizer.draw_dataset_dict(d)
  plt.imshow(vis.get_image()[:, :, ::-1])
  plt.show()


# Définition d'une classe CocoTrainer qui hérite de DefaultTrainer
class CocoTrainer(DefaultTrainer):
  # Définition d'une méthode de classe pour construire un évaluateur pour évaluer les performances du modèle entraîné selon "cfg" et "dataset_name"
  @classmethod
  def build_evaluator(cls, cfg, dataset_name, output_folder=None):

    if output_folder is None:
        os.makedirs("coco_eval", exist_ok=True)
        output_folder = "coco_eval"

    # Retourne un nouvel objet COCOEvaluator qui sera utilisé pour évaluer les performances du modèle sur l'ensemble de données spécifié
    # Le paramètre 'False' signifie que l'évaluateur n'est pas en mode d'évaluation distribuée. (l’évaluation est effectuée sur un seul processeur)
    return COCOEvaluator(dataset_name, cfg, False, output_folder)
###################################################################################################

# Obtention de la configuration par défaut de Detectron2
cfg = get_cfg()
# Définition du périphérique sur lequel le modèle sera entraîné
# "cuda" signifie que le modèle sera entraîné sur un GPU si disponible
if torch.cuda.is_available():
        cfg.MODEL.DEVICE = 'cuda'
else :
        cfg.MODEL.DEVICE = 'cpu'
# Définition du répertoire où où les résultats de l'entraînement seront sauvegardés
cfg.OUTPUT_DIR = "data/projet_m1/models/Detectron2_Models7"
#fusion des paramètres de configuration du modèle à partir du fichier YAML 
#Ce fichier contient les paramètres préconfigurés pour un modèle spécifique de détection d'instances appelé Mask R-CNN avec le backbone X-101-32x8d FPN
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("my_dataset_train",)# noms des ensembles de données utilisés pour l'entraînement
cfg.DATASETS.TEST = ("my_dataset_val",) # noms des ensembles de données utilisés pour la validation
# Définition du nombre de threads pour charger les données en parallèle pendant l'entraînement 
#En le réglant sur 0, cela désactive le chargement parallèle et charge les données séquentiellement
cfg.DATALOADER.NUM_WORKERS = 0
# Si "True", les images sans annotations seront filtrées et non utilisées pendant l'entraînement
cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS=False
# Si "True", le modèle prédit également des masques en plus des boîtes englobantes
cfg.MODEL.MASK_ON = True
# Définition de la résolution du pooler pour la tête de la boîte ROI et la tête du masque ROI
cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION = 20
cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION = 20
# Définition des tailles des ancres générées pour la détection d'objets
cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[8], [16], [32], [64], [128]]
# Définition de la fraction des ROI positives dans un mini-lot
cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.5 
# Chargement des poids du modèle à partir des poids d'un modèle pré-entraîné sur COCO
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
# Définition du nombre d'images par lot et le taux d'apprentissage de base.
cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.BASE_LR = 0.001 
# Définition du nombre maximum d'itérations d'entraînement et la période de sauvegarde des points de contrôle
cfg.SOLVER.MAX_ITER = 5000
cfg.SOLVER.CHECKPOINT_PERIOD = 1000
# Définition des étapes où le taux d'apprentissage doit être réduit qui est vide => le taux d'apprentissage reste constant
cfg.SOLVER.STEPS = []
# Définition de la taille de lot par image et le nombre de classes
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
# Définition du seuil de score pour les prédictions lors des tests
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.15
# Si True, une normalisation précise du lot est effectuée lors des tests
cfg.TEST.PRECISE_BN.ENABLED = True
cfg.TEST.PRECISE_BN.NUM_ITER = 1000
# Définition du seuil NMS lors des tests et le nombre maximum de détections par image
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST= 0.59
cfg.TEST.DETECTIONS_PER_IMAGE = 1000
# Définition de la période d'évaluation pendant l'entraînement
cfg.TEST.EVAL_PERIOD = 159
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
# Création d'un entraîneur avec la configuration spécifiée et commencement de l'entraînement
trainer = CocoTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()


cfg.MODEL.WEIGHTS = "data/projet_m1/models/Detectron2_Models6/model_final.pth"
# Mettre à jour le seuil de score pour les prédictions lors des tests
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2

cfg.TEST.PRECISE_BN.ENABLED = True
cfg.TEST.PRECISE_BN.NUM_ITER = 1000

# Mettre à jour le seuil NMS lors des tests et le nombre maximum de détections par image
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST= 0.6
cfg.TEST.DETECTIONS_PER_IMAGE = 1000

predictor = DefaultPredictor(cfg)

# Création d'un évaluateur pour l'ensemble de données de test et effectuation d'une inférence sur l'ensemble de données de test
evaluator = COCOEvaluator("my_dataset_test", cfg, False, output_dir="./output/",max_dets_per_image=1000)
val_loader = build_detection_test_loader(cfg, "my_dataset_test")
inference_on_dataset(trainer.model, val_loader, evaluator)

#####################################################################################################



from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import ColorMode



# Sélection d'un échantillon aléatoire de l'ensemble de données de test
for d in random.sample(test_dataset_dicts, 1):
    #im = cv2.imread(d["file_name"])
    im = cv2.imread("projetm1_data/test/Tbaro-WT-pyruvte-point-final_5C_ch00_part_2_jpg.rf.b0760c8d5513cf05cdb63560378862b4.jpg")
   
    # Utilisation du prédicteur pour faire des prédictions sur l'image
    outputs = predictor(im)
    instances = outputs["instances"]
   
    # Création d'un objet Visualizer pour visualiser les prédictions sur l'image
    v = Visualizer(im[:, :, ::-1],
                   metadata=test_metadata,
                   scale=1,
                   instance_mode=ColorMode.IMAGE
    )
    
    # Dessine les prédictions d'instance sur l'image
    v = v.draw_instance_predictions(instances.to("cpu"))
    
    img = v.get_image()[:,:,[2,1,0]]
    img = Image.fromarray(img)
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    #out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    # Comptage de nombre de boîtes englobantes prédites et affichage
    num_boxes = len(instances.pred_boxes)
    print("Nombre de boîtes englobantes prédites :", num_boxes)

    input_filename = "data/projet_m1/input_image.jpg"
    cv2.imwrite(input_filename, im)


    output_filename = "data/projet_m1/output_image.jpg"
    cv2.imwrite(output_filename, v.get_image())

    print(f"Images sauvegardées : {input_filename}, {output_filename}")



################################################################################################
from detectron2.structures import Boxes, BoxMode,pairwise_ioa

# Initialisation des listes pour stocker les valeurs d'Intersection sur Union (IoU) et les compteurs
ioa_values = []
num_images_with_annotations = 0

# Boucle sur chaque image dans l'ensemble de données de test
for d in test_dataset_dicts:
    im = cv2.imread(d["file_name"])
    
    # Prédiction des instances sur l'image
    outputs = predictor(im)

    # Si l'image n'a pas d'annotations ou si aucune instance n'a été prédite, passe à l'image suivante
    if "annotations" not in d or len(d["annotations"]) == 0 or len(outputs["instances"]) == 0:
        continue  

    num_images_with_annotations += 1

    # Conversion des boîtes englobantes des annotations en coordonnées XYXY_ABS et création d'un objet Boxes
    bboxes_gt = [
        BoxMode.convert(obj["bbox"], BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
        for obj in d["annotations"]
    ]
    bboxes_gt = Boxes(torch.Tensor(bboxes_gt))

    # Récupération des boîtes englobantes prédites
    bboxes_pred = outputs["instances"].pred_boxes
    
    # Conversion des boîtes englobantes réelles au même type de périphérique que les boîtes prédites
    bboxes_gt = bboxes_gt.to(bboxes_pred.device)

    # Tri des indices des boîtes englobantes prédites et réelles
    sorted_indices_pred = sorted(range(len(bboxes_pred)), key=lambda i: tuple(bboxes_pred.tensor[i]))
    sorted_indices_gt = sorted(range(len(bboxes_gt)), key=lambda i: tuple(bboxes_gt.tensor[i]))

    # Création de nouvelles listes de boîtes englobantes triées
    sorted_bboxes_pred = torch.cat([bboxes_pred[i].tensor for i in sorted_indices_pred])
    sorted_bboxes_gt = torch.cat([bboxes_gt[i].tensor for i in sorted_indices_gt])

    # Calcul de l'Intersection sur l'Union (IoU) pour chaque paire de boîtes englobantes réelle et prédite
    IOUs = pairwise_ioa(bboxes_gt, bboxes_pred)

    # Récupération de la valeur IoU maximale pour chaque boîte englobante réelle
    max_values, _ = torch.max(IOUs, dim=1)

    # Calcul de la valeur IoU moyenne pour l'image
    average_max_value = torch.mean(max_values)
    ioa_values.append(average_max_value.item())

# Calcul de la valeur IoU moyenne sur toutes les images avec des annotations
if num_images_with_annotations > 0:
    average_iou = sum(ioa_values) / num_images_with_annotations
    print("Average IoU across images with annotations:", average_iou)
else:
    print("No images with annotations or predictions found.")

# Définition du seuil IoA pour considérer une prédiction comme un vrai positif
ioa_threshold = 0.5
# Initialisation des listes pour stocker les valeurs de rappel et de précision
recall_values = []
precison_values = []
# Réinitialisation du compteur du nombre d'images avec des annotations
num_images_with_annotations = 0

# Boucle sur chaque image dans l'ensemble de données de test (même processus que précédemment)
for d in test_dataset_dicts:
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)

    if "annotations" not in d or len(d["annotations"]) == 0 or len(outputs["instances"]) == 0:
        continue 

    num_images_with_annotations += 1

    bboxes_gt = [
        BoxMode.convert(obj["bbox"], BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
        for obj in d["annotations"]
    ]
    bboxes_gt = Boxes(torch.Tensor(bboxes_gt))
    bboxes_pred = outputs["instances"].pred_boxes
    bboxes_gt = bboxes_gt.to(bboxes_pred.device)

    sorted_indices_pred = sorted(range(len(bboxes_pred)), key=lambda i: tuple(bboxes_pred.tensor[i]))
    sorted_indices_gt = sorted(range(len(bboxes_gt)), key=lambda i: tuple(bboxes_gt.tensor[i]))

    sorted_bboxes_pred = torch.cat([bboxes_pred[i].tensor for i in sorted_indices_pred])
    sorted_bboxes_gt = torch.cat([bboxes_gt[i].tensor for i in sorted_indices_gt])

    IOUs = pairwise_ioa(bboxes_pred,bboxes_gt)
    max_values, _ = torch.max(IOUs, dim=1)
    
    # Calcul du nombre de vrais positifs (prédictions avec une valeur IoA supérieure au seuil)
    true_positives = max_values >= ioa_threshold
    
    # Calcul du rappel et de la précision pour l'image
    recall = true_positives.float().sum() / float(len(bboxes_gt))
    recall_values.append(recall.item())
    precision = true_positives.float().sum() / float(len(bboxes_pred))
    precison_values.append(precision.item())

# Calcul du rappel et de la précision moyens sur toutes les images
average_recall = sum(recall_values) / len(recall_values)
print("Average recall: ", round(average_recall,2))
average_precision = sum(precison_values) / len(precison_values)
print("Average precision:", round(average_precision,2))


