import os
import cv2

def diviser_en_8_parts(image_path, output_folder):
    """
    Cette fonction divise chaque image en 8 parties égales et les enregistre dans un dossier de sortie.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Impossible de charger l'image à partir de {image_path}")
        return

    # On obtient les dimensions de l'image.
    height, width, _ = image.shape
    part_height = height // 2
    part_width = width // 4

    counter = 1  
    # On parcoure les 8 parties de l'image.
    for row in range(2):
        for col in range(4):
            # On découpe l'image en 8 parties et on redimensionne chaque partie.
            part = image[row * part_height:(row + 1) * part_height, col * part_width:(col + 1) * part_width]
            
            # On nomme et on enregistre chaque partie de l'image.
            part_name = f"{os.path.splitext(os.path.basename(image_path))[0]}_part_{counter}.jpg"
            part_path = os.path.join(output_folder, part_name)
            cv2.imwrite(part_path, part)
            counter += 1


input_folder = "C:/Pour Yann"
output_folder = "C:/images_decoupees_8"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for image_filename in os.listdir(input_folder):
    if image_filename.endswith(".jpg") or image_filename.endswith(".png"):
        image_path = os.path.join(input_folder, image_filename)
        diviser_en_8_parts(image_path, output_folder)

