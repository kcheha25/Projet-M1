from flask import Flask, render_template, request
import os
from DeepLearning import inference_detectron2 ,delete_files_in_folder
#Webserver gateway interface
app = Flask (__name__)
#On définit les chemins de sauvegarde des 2 images téléversées.
BASE_PATH = os.getcwd()
UPLOAD_PATH1 = os.path.join(BASE_PATH, 'static/upload1/')
UPLOAD_PATH2 = os.path.join(BASE_PATH, 'static/upload2/')


@app.route('/', methods=['POST', 'GET'])
def index():

    result = None  

    if request.method == 'POST':
        
        #On supprime les images présents dans les dossiers, UPLOAD_PATH1 et UPLOAD_PATH2, 
        #lors des précédents téléversement.
        delete_files_in_folder(UPLOAD_PATH1)
        delete_files_in_folder(UPLOAD_PATH2)
        
        #On récupère les images choisies par utilisateur à partir des champs 'fileInput1' et 'fileInput2'
        #du formulaire.
        #On sauvegarde les images dans leurs dossiers spécifiques.
        upload_file1 = request.files['fileInput1']
        filename1 = upload_file1.filename
        path_save1 = os.path.join(UPLOAD_PATH1, filename1)
        upload_file1.save(path_save1)
        upload_file2 = request.files['fileInput2']
        filename2 = upload_file2.filename
        path_save2 = os.path.join(UPLOAD_PATH2, filename2)
        upload_file2.save(path_save2)
        
        #On appel la fonction inference_detectron2() pour effectuer l'inférence sur les images
        #téléversés et on stocke le résultat dans la variable result.
        result = inference_detectron2(path_save1,path_save2)
    
    #la fonction render_template() charge le modèle HTML index.html et 
    #y injecte les données résultantes de l'opération d'inférence avant de renvoyer 
    #la page web au navigateur web pour l’affichage.
    return render_template('index.html', result=result)

if __name__ =="__main__":
    app.run(debug=True)