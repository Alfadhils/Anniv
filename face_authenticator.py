import matplotlib.pyplot as plt
import os
import cv2
from deepface import DeepFace

class FaceAuthenticator:
    def __init__(self, db_path, recreate=False, model_name='Facenet', backend='retinaface'):
        self.db_path = db_path
        self.model_name = model_name
        self.backend = backend

        self.faces = self.validate_db(recreate)

    def predict(self, img_path, plot=True):
        face = self.validate_image(img_path)

        df = DeepFace.find(img_path=img_path, db_path=self.db_path, detector_backend=self.backend, model_name=self.model_name, silent=True)[0]
        score = len(df)/(len(self.faces)) * 100

        if plot :
            if len(face) > 0:
                plt.imshow(face[0]['face'])
                plt.title('Score : {:.2f}%'.format(score))
            else:
                print('No face detected')
        
        return score > 15

    def validate_db(self, recreate):

        if recreate:
            self.remove_rep()
            sample_img = os.path.join(self.db_path, os.listdir(self.db_path)[0])
            print('Create representation file')
            _ = DeepFace.find(img_path=sample_img, db_path=self.db_path, detector_backend=self.backend, model_name=self.model_name, silent=True)[0]
        
        faces = []
        for img_db in os.listdir(self.db_path):
            if img_db.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                img_path = os.path.join(self.db_path, img_db)
                img = cv2.imread(img_path)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_resized = cv2.resize(img_rgb, (228,228))
                faces.append(img_resized)
                
        print('Database ready to use')
        return faces
    
    def remove_rep(self):
        representation_path = os.path.join(self.db_path, 'representations_facenet.pkl')
        if os.path.exists(representation_path):
            os.remove(representation_path)
            print('Representation file removed') 

    def validate_image(self, img_path):
        try :
            face = DeepFace.extract_faces(img_path, detector_backend=self.backend)
        except Exception as e:
            print(f'Error: {e}')
            return
        
        return face
    
    def plot_faces(self):
        fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(14, 6))
        for i, face in enumerate(self.faces):
            axes.flatten()[i].imshow(face)

