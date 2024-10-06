from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
import torch
from detectron2.utils.logger import setup_logger
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import ColorMode
import distutils.core
import cv2
import os
import numpy as np

setup_logger()

dist = distutils.core.run_setup("setup.py")

def inference_detectron2(image_path1,image_path2):
    """
   Cette fonction effectue une inférence à l'aide du modèle Detectron2 sur les deux images.

   """
    output_folder1 = "static/upload1"
    output_folder2 = "static/upload2"
    # Dans ce qui suit la configuration des hyperparamètres et des poids sauvegarder
    # pour le modèle Detectron2.
    cfg = get_cfg()
    if torch.cuda.is_available():
        cfg.MODEL.DEVICE = 'cuda'
    else :
        cfg.MODEL.DEVICE = 'cpu'
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))

    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS=False
    cfg.MODEL.MASK_ON = True
    cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION = 20
    cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION = 20
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[8], [16], [32], [64], [128]]
    cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.5
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 0.001 
    cfg.SOLVER.MAX_ITER = 5000
    cfg.SOLVER.CHECKPOINT_PERIOD = 1000
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    
    cfg.MODEL.WEIGHTS = "data/projet_m1/models/Detectron2_Models6/model_final.pth"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.15
    cfg.TEST.PRECISE_BN.ENABLED = True
    cfg.TEST.PRECISE_BN.NUM_ITER = 1000
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST= 0.59
    cfg.TEST.DETECTIONS_PER_IMAGE = 1000
    
    # On initialise le prédicteur de notre modèle.
    predictor = DefaultPredictor(cfg)
    
    # On exécute l'inférence sur chacune des deux images à l’aide de la fonction infer().
    image1,out1,num1= infer(image_path1,output_folder1,predictor)
    image2,out2,num2= infer(image_path2,output_folder2,predictor)
    
    # On calcul la différence entre les nombres de boîtes prédites (point final – vide) 
    # et la concentration cellulaire.
    num3 = num2 - num1
    num4 = round(round(((10**6) * num2)/1.7)*10**-8,8)
    
    #Les résultats sont sous forme de dictionnaire contenant les chemins des images
    # résultantes et les nombres de boîtes prédites.
    return {
    'image_path1': image1,
    'assembled_image_path1': out1,
    'image_path2': image2,
    'assembled_image_path2': out2 ,
    'num1':num1,
    'num2':num2,
    'num3':num3,
    'num4':num4}

def infer(image_path,output_folder,predictor):
     """
       Cette fonctio effectue l'inférence à l'aide du modèle Detectron2 sur une seule image.

     """
     image = cv2.imread(image_path)
     if image is None:
         print(f"Impossible de charger l'image à partir de {image_path}")
         return None, None
     
     # On récupère les dimensions de l'image.
     height, width, _ = image.shape
     part_height = height // 2
     part_width = width // 4
     counter = 1  
     parts_with_annotations = []
     
     
     if not os.path.exists(output_folder):
         os.makedirs(output_folder)
      
     # La base de données de test est enregistrée.
     if "my_dataset_test" not in DatasetCatalog.list(): 
         register_coco_instances("my_dataset_test", {}, "C:/Users/karim/Desktop/projetm1_data/test/_annotations.coco.json", "C:/Users/karim/Desktop/projetm1_data/test")
     test_metadata = MetadataCatalog.get("my_dataset_test")
     
     
     num_boxes=0
     for row in range(2):
         for col in range(4):
             #On découpe l'image en 8 parties et on redimensionne chaque partie.
             part = image[row * part_height:(row + 1) * part_height, col * part_width:(col + 1) * part_width]
             resized_part = cv2.resize(part, (640, 640))
             
             # On réalise le Prétraitement nécessaire sur chaque partie à l’aide
             # de la fonction preprocess_image().
             preprocessed_image_path = os.path.join(output_folder, f"part_{counter}_preprocessed.jpg")
             cv2.imwrite(preprocessed_image_path, resized_part)
             preprocessed_image_path = preprocess_image(preprocessed_image_path,output_folder)
             preprocessed_part = cv2.imread(preprocessed_image_path)
             
             # On effectue l'inférence sur chaque partie.
             outputs = predictor(preprocessed_part)
             part=cv2.resize(part,(320,640))
             v = Visualizer(part[:, :, ::-1],
                             metadata=test_metadata,
                             scale=1,
                             instance_mode=ColorMode.IMAGE
             )
             out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
             annotated_image = out.get_image()[:, :, [2, 1, 0]]
             
             # On enregistre l'image annotée et sauvegarde dans num_boxes le nombre de cellules prédites.
             part_name = f"{os.path.splitext(os.path.basename(image_path))[0]}part{counter}.jpg"
             part_path = os.path.join(output_folder, part_name)
             num_boxes += len(outputs["instances"].pred_boxes)
             cv2.imwrite(part_path, annotated_image)
             if os.path.exists(part_path):  
                 parts_with_annotations.append(part_path)
             else:
                 print(f"La partie {part_name} n'a pas été correctement enregistrée.")
             counter += 1    
     
    # On réassemble les 8 parties annotées en une seule image
     assembled_image = np.zeros((2 * part_height, 4 * part_width, 3), dtype=np.uint8)
     for row in range(2):
         for col in range(4):
             part_idx = row * 4 + col
             if part_idx < len(parts_with_annotations):
                  part_path = parts_with_annotations[part_idx]
                  part_image = cv2.imread(part_path)
                  if part_image is not None:  
                     part_image_resized = cv2.resize(part_image, (part_width, part_height))
                     assembled_image[row * part_height:(row + 1) * part_height, col * part_width:(col + 1) * part_width] = part_image_resized
                  else:
                       print(f"Impossible de lire l'image {part_path}.")
             else:
                     print("Certaines parties annotées sont manquantes.")
         
     # On enregistre l'image assemblée. 
     assembled_image_path = os.path.join( output_folder, "assembled_image.jpg")
     cv2.imwrite(assembled_image_path, assembled_image)  

    # Le chemin de l'image d'origine, le chemin de l'image assemblée avec les annotations, 
    # et le nombre de boîtes prédites sont renvoyés.     
     return image_path, assembled_image_path,num_boxes


def preprocess_image(image_path,output_folder):
    """
    Cette fonction constitue la phase de prétraitement effectuée sur nos images.   
    
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
    normalized_image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    # On applique le filtre bilatéral.
    blurred_image = cv2.bilateralFilter(normalized_image, 5, 75, 75)
    # Ensuite on applique une égalisation d'histogramme adaptative (CLAHE) 
    # sur l'image résultante du filtre bilatéral.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized_image = clahe.apply(blurred_image)
    # On applique un filtre de renforcement de netteté sur l’image résultante de l’étape précédente.
    sharp_kernel = np.array([[-1, -1, -1],
                             [-1,  9, -1],
                             [-1, -1, -1]])
    sharpened_image = cv2.filter2D(equalized_image, -1, sharp_kernel)
    #On sauvegarde de l'image prétraitée après avoir redimensionner l’image selon 
    # les dimensions similaires aux données d’entraînement du modèle.
    preprocessed_path = os.path.join(output_folder, f"preprocessed_{os.path.basename(image_path)}")
    sharpened_image = cv2.resize(sharpened_image, (320, 640))
    cv2.imwrite(preprocessed_path, sharpened_image)
    
    # Le chemin de l'image prétraitée est renvoyé.
    return preprocessed_path
    


def delete_files_in_folder(folder_path):
    """
    Cette fonction supprime tous les fichiers présents dans un dossier spécifié.
    """
    try:
        files = os.listdir(folder_path)
        
        for file_name in files:
            file_path = os.path.join(folder_path, file_name)
            
            if os.path.isfile(file_path):
                
                os.remove(file_path)
                print(f"Deleted: {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")


