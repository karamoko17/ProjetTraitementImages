import cv2
import tkinter as tk
import numpy as np
import random
from PIL import Image, ImageTk

images_btn = []

def create_image_button(frame, image_path, width, height, action, row, column, padx):
        img = Image.open(image_path)
        img.thumbnail((width, height))
        photo = ImageTk.PhotoImage(img)
        images_btn.append(photo)
        btn = tk.Button(frame, image=photo, command=action, bd=0, relief=tk.FLAT)
        btn.grid(row=row, column=column, padx=padx)
        return btn

class App:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        # Label pour afficher les transformations actuelles
        self.current_filters_label = tk.Label(self.window, text="", font=("Arial", 14))
        self.current_filters_label.pack()

        # Initialisation de la capture vidéo à partir de la webcam
        self.vid = cv2.VideoCapture(0)

        # Paramétrage des dimensions de la vidéo capturée
        self.vid.set(cv2.CAP_PROP_FRAME_WIDTH, 563)
        self.vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 612)

        # Création d'un canevas pour afficher la vidéo
        self.canvas = tk.Canvas(window, width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH), height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.canvas.pack()

        # Création d'un menu interactif
        self.create_menu()

        # Drapeaux pour indiquer quels traitements doivent être appliqués
        self.apply_sepia = False
        self.overlay_snowflakes = False
        self.change_background = False
        self.detect_eye_flag = False
        self.detect_nose_flag = False
        self.detect_barbe_flag = False
        
        # Lancement du processus de mise à jour de la capture vidéo
        self.update()

        # Gestion de la fermeture de la fenêtre
        self.window.protocol("WM_DELETE_WINDOW", self.on_close)

    def update_current_filters_label(self):
        filters_applied = {
            "apply_sepia": "sepia",
            "overlay_snowflakes": "neige",
            "change_background": "plage",
            "detect_eye_flag": "lunettes",
            "detect_nose_flag": "chien",
            "detect_barbe_flag": "barbe"
        }

        current_filters = [filters_applied[key] for key in filters_applied if getattr(self, key)]
        self.current_filters_label.config(text=f"{', '.join(current_filters)}")
        
    def apply_sepia(self):
        self.apply_sepia = not self.apply_sepia

    def overlay_snowflakes(self):
        self.overlay_snowflakes = not self.overlay_snowflakes

    def change_background(self):
        self.change_background = not self.change_background
    
    def detect_eye(self):
        self.detect_eye_flag = not self.detect_eye_flag

    def detect_nose(self):
        self.detect_nose_flag = not self.detect_nose_flag
        
    def detect_barbe(self):
        self.detect_barbe_flag = not self.detect_barbe_flag
    
    def apply_all(self):
        self.apply_sepia = True
        self.overlay_snowflakes = True
        self.change_background = True
        self.detect_eye_flag = True
        self.detect_nose_flag = True
        self.detect_barbe_flag = True

    def surprise(self):
        self.stop_all()
        transformations = [
            "apply_sepia",
            "overlay_snowflakes",
            "change_background",
            "detect_eye_flag",
            "detect_nose_flag",
            "detect_barbe_flag"
        ]
        random_transformation = random.choice(transformations)
        setattr(self, random_transformation, True)
    
    def stop_all(self):
        self.apply_sepia = False
        self.overlay_snowflakes = False
        self.change_background = False
        self.detect_eye_flag = False
        self.detect_nose_flag = False
        self.detect_barbe_flag = False

    def create_menu(self):
        menu_frame = tk.Frame(self.window)
        menu_frame.pack(pady = 10)

        # Création des boutons :
        #       - Changer le fond
        #       - Filtre Sépia
        #       - Flocons de neige
        #       - Barbe
        #       - Chien
        #       - Lunettes
        #       - Appliquer tout

        btn_width = 50
        btn_height = 50

        create_image_button(menu_frame, "images/beach.png", btn_width, btn_height, self.change_background, row=0, column=0, padx=10)
        create_image_button(menu_frame, "images/sepia.png", btn_width, btn_height, self.apply_sepia, row=0, column=1, padx=10)
        create_image_button(menu_frame, "images/snowflakes.png", btn_width, btn_height, self.overlay_snowflakes, row=0, column=2, padx=10)
        create_image_button(menu_frame, "images/barbe.png", btn_width, btn_height, self.detect_barbe, row=0, column=3, padx=10)
        create_image_button(menu_frame, "images/chien.png", btn_width, btn_height, self.detect_nose, row=0, column=4, padx=10)
        create_image_button(menu_frame, "images/lunettes.png", btn_width, btn_height, self.detect_eye, row=0, column=5, padx=10)
        create_image_button(menu_frame, "images/all.png", btn_width, btn_height, self.apply_all, row=0, column=7, padx=10)
        create_image_button(menu_frame, "images/cross.png", btn_width, btn_height, self.stop_all, row=0, column=9, padx=10)
        create_image_button(menu_frame, "images/surprise.png", btn_width, btn_height, self.surprise, row=0, column=8, padx=10)

    def update(self):
        # Capture de la trame vidéo
        ret, frame = self.vid.read()
        virgin_frame = frame.copy()

        # Application des traitements choisis
        if self.change_background:
            frame = self.change_background_function(frame, cv2.imread("images/fond-563x612.jpg"))
        if self.overlay_snowflakes:
            frame = self.overlay_snowflakes_effect(virgin_frame)
        if self.detect_eye_flag:
            frame = self.detect_eye_function(frame)
        if self.detect_nose_flag:
            frame = self.detect_nose_function(frame)
        if self.detect_barbe_flag:
            frame = self.detect_barbe_function(frame)
        if self.apply_sepia:
            frame = self.apply_sepia_filter(frame)

        # Mise à jour du canva avec la nouvelle image
        self.photo = ImageTk.PhotoImage(image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        self.canvas.create_image(0, 0, image = self.photo, anchor = tk.NW)

        # Mise à jour du label des filtres en cours
        self.update_current_filters_label()

        # Appel récursif périodique
        self.window.after(10, self.update)

    def apply_sepia_filter(self, frame):
        sepia_matrix = np.array([
            [0.131, 0.534, 0.272],
            [0.168, 0.686, 0.349],
            [0.189, 0.769, 0.393]
        ])
        sepia_frame = cv2.transform(frame, sepia_matrix)
        sepia_frame = np.clip(sepia_frame, 0, 255).astype(np.uint8)
        
        return sepia_frame

    def change_background_function(self, sujet, fond):
        height, width, _ = sujet.shape
        fond = cv2.resize(fond, (width, height))

        lower_white = np.array([160, 160, 160])
        upper_white = np.array([255, 255, 255])
        mask_white = cv2.inRange(sujet, lower_white, upper_white)

        mask_inv = cv2.bitwise_not(mask_white)

        result = cv2.bitwise_and(sujet, sujet, mask=mask_inv)
        result += cv2.bitwise_and(fond, fond, mask=mask_white)

        return result
    
    def overlay_snowflakes_effect(self, frame):
        num_snowflakes = 75

        lower_white = np.array([160, 160, 160])
        upper_white = np.array([255, 255, 255])
        mask_white = cv2.inRange(frame, lower_white, upper_white)

        white_pixels = cv2.findNonZero(mask_white)

        '''
        if self.change_background:
            mask_inv = cv2.bitwise_not(mask_white)

            frame = cv2.bitwise_and(frame, frame, mask=mask_inv)
            frame += cv2.bitwise_and(cv2.imread("images/fond-563x612.jpg"),  cv2.imread("images/fond-563x612.jpg"), mask=mask_white)
        '''

        if white_pixels is not None:
                for _ in range(num_snowflakes):
                    rand_pixel = white_pixels[np.random.randint(0, len(white_pixels))][0]
                    x, y = rand_pixel[0], rand_pixel[1]
                    frame[y:y + 4, x:x + 4] = [255, 255, 255]

        return frame

    def detect_eye_function(self, image):
        # Conversion en niveaux de gris
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Utilisez un classificateur Haar pour détecter la bouche
        eye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye_tree_eyeglasses.xml')
       
        # Détecter les yeux dans l'image
        eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Chargement de l'image par défaut pour les lunettes
        default_glasses_image = cv2.imread('images/lunettes.png', cv2.IMREAD_UNCHANGED)
        # Load the cascade
        face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')
        # Détection des visages
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
        
        # Dessiner l'image par défaut des lunettes dans les zones détectées
        for (x, y, w, h) in faces:
            # Région d'intérêt (ROI) pour les yeux dans le visage détecté
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = image[y:y + h, x:x + w]

            # Détection des yeux dans la ROI
            eyes_in_roi = eye_cascade.detectMultiScale(roi_gray)

            # Stocker les coordonnées des yeux
            eye_coords = []

            # Pour chaque paire d'yeux détectée
            for (ex, ey, ew, eh) in eyes_in_roi:
                eye_coords.append((ex, ey, ew, eh))

            # Si deux yeux sont détectés
            if len(eye_coords) == 2:
                # Calculer la position moyenne des deux yeux
                avg_x = int((eye_coords[0][0] + eye_coords[1][0]) / 2)
                avg_y = int((eye_coords[0][1] + eye_coords[1][1]) / 2)
                avg_w = int((eye_coords[0][2] + eye_coords[1][2]) / 2)
                avg_h = int((eye_coords[0][3] + eye_coords[1][3]) / 2)

                # Redimensionner les lunettes à la taille des deux yeux
                resized_sunglasses = cv2.resize(default_glasses_image, (avg_w * 3, avg_h * 2))

                # Superposer les lunettes sur l'image
                x_offset, y_offset = avg_x - avg_w, avg_y - avg_h // 2  # Ajuster la position des lunettes
                y1, y2 = y_offset, y_offset + resized_sunglasses.shape[0]
                x1, x2 = x_offset, x_offset + resized_sunglasses.shape[1]
            

                # Région de l'image pour les lunettes
                alpha_s = resized_sunglasses[:, :, 3] / 255.0
                alpha_l = 1.0 - alpha_s
                for c in range(0, 3):
                    roi_color[y1:y2, x1:x2, c] = (alpha_s * resized_sunglasses[:, :, c] + alpha_l * roi_color[y1:y2, x1:x2, c])
                
        return image
      
    def detect_nose_function(self, image):
        # Conversion en niveaux de gris
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Utilisez un classificateur Haar pour détecter le nez
        nose_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_mcs_nose.xml')
       
       # Charger l'image par défaut pour le nez de chien
        default_nose_image = cv2.imread('images/chien.png', cv2.IMREAD_UNCHANGED)

        # Load the cascade
        face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')
        # Détection des visages
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
    
        
        # Ajuster cette valeur pour contrôler la position verticale du nez
        offset_y_percentage = 0.15

        for (x, y, w, h) in faces:
            # Ajuster la taille de l'image du nez pour mieux s'adapter à l'ensemble du visage
            resized_nose = cv2.resize(default_nose_image, (w, h))

            # Région d'intérêt (ROI) pour le nez dans le visage détecté
            roi_color = image[y:y + h, x:x + w]

            # Assurer que les coordonnées sont valides
            if 0 <= y < image.shape[0] and 0 <= y + h < image.shape[0] and 0 <= x < image.shape[1] and 0 <= x + w < image.shape[1]:
                # Région de l'image pour le nez
                alpha_n = resized_nose[:, :, 3] / 255.0
                alpha_l = 1.0 - alpha_n

                # Déplacer l'image du nez vers le haut en ajustant les coordonnées y
                offset_y = int(h * offset_y_percentage)
                y_offset = y - offset_y if y - offset_y >= 0 else 0

                for c in range(0, 3):
                    # Appliquer l'image du nez sur l'ensemble du visage
                    image[y_offset:y_offset + h, x:x + w, c] = (alpha_n * resized_nose[:, :, c] + alpha_l * image[y_offset:y_offset + h, x:x + w, c])
        return image   
    
    def detect_barbe_function(self, image):
        # Conversion en niveaux de gris
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

       # Charger l'image par défaut pour le nez de chien
        default_nose_image = cv2.imread('images/barbe.png', cv2.IMREAD_UNCHANGED)

        # Load the cascade
        face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')
        # Détection des visages
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
    
        
        # Ajuster cette valeur pour contrôler la position verticale du nez vers le bas
        offset_y_percentage = 0.3 #0.45

        for (x, y, w, h) in faces:
            # Ajuster la taille de l'image du nez pour mieux s'adapter à l'ensemble du visage
            resized_nose = cv2.resize(default_nose_image, (w, h))

            # Région d'intérêt (ROI) pour le nez dans le visage détecté
            roi_color = image[y:y + h, x:x + w]

            # Assurer que les coordonnées sont valides
            if 0 <= y < image.shape[0] and 0 <= y + h < image.shape[0] and 0 <= x < image.shape[1] and 0 <= x + w < image.shape[1]:
                # Région de l'image pour le nez
                alpha_n = resized_nose[:, :, 3] / 255.0
                alpha_l = 1.0 - alpha_n

                # Déplacer l'image du nez vers le bas en ajustant les coordonnées y
                offset_y = int(h * offset_y_percentage)
                y_offset = y + offset_y if y + h + offset_y <= image.shape[0] else y

                for c in range(0, 3):
                    # Appliquer l'image du nez sur l'ensemble du visage
                    image[y_offset:y_offset + h, x:x + w, c] = (alpha_n * resized_nose[:, :, c] + alpha_l * image[y_offset:y_offset + h, x:x + w, c])
        
        return image   

    def on_close(self):
        if self.vid.isOpened():
            self.vid.release()
        self.window.destroy()


# Création de l'application
root = tk.Tk()
app = App(root, "App")
root.mainloop()