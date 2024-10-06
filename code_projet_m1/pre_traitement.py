import cv2
import numpy as np
import os

def preprocess_image(image_path):
    # Lecture de l'image en niveaux de gris à partir de "image_path"
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Normalisation de l'image pour étendre sa plage de valeurs entre 0 et 255
    normalized_image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    
    # Application d'un flou bilatéral pour réduire le bruit tout en préservant les contours
    blurred_image = cv2.bilateralFilter(normalized_image, 5, 75, 75)
    
    # Création et application d'un objet CLAHE (Contrast Limited Adaptive Histogram Equalization) pour améliorer le contraste de l'image
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized_image = clahe.apply(blurred_image)
    
    # Définition d'un noyau de netteté
    sharp_kernel = np.array([[-1, -1, -1],
                             [-1,  9, -1],
                             [-1, -1, -1]])
    # Application du noyau de netteté à "equalized_image" en les convoluant
    sharpened_image = cv2.filter2D(equalized_image, -1, sharp_kernel)
    
    
    return sharpened_image


def preprocess_images_in_folder(input_folder, output_folder):
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    
    for filename in os.listdir(input_folder):
        if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
            
            input_image_path = os.path.join(input_folder, filename)
            
            # Prétraitement de l'image
            preprocessed_image = preprocess_image(input_image_path)
            
            
            output_image_path = os.path.join(output_folder, filename)
            
            # Sauvegarder l'image prétraitée dans le dossier de sortie
            cv2.imwrite(output_image_path, preprocessed_image)
            print(f"Image prétraitée sauvegardée: {output_image_path}")





input_folder ='C:/train'


output_folder = 'C:/output_images'

preprocess_images_in_folder(input_folder, output_folder)

