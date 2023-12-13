import cv2
import tkinter as tk
import numpy as np
from PIL import Image, ImageTk, ImageFilter

class App:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

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

        tk.Button(menu_frame, text = "Changer le fond", command = self.change_background).grid(row = 0, column = 0, padx = 10)
        tk.Button(menu_frame, text = "Filtre Sépia", command = self.apply_sepia).grid(row = 0, column = 1, padx = 10)
        tk.Button(menu_frame, text = "Flocons de neige", command = self.overlay_snowflakes).grid(row = 0, column = 2, padx = 10)
        tk.Button(menu_frame, text = "Barbe", command = self.detect_barbe).grid(row = 0, column = 3, padx = 10)
        tk.Button(menu_frame, text = "Chien", command = self.detect_nose).grid(row = 0, column = 4, padx = 10)
        tk.Button(menu_frame, text = "Lunettes", command = self.detect_eye).grid(row = 0, column = 5, padx = 10)
        tk.Button(menu_frame, text = "Appliquer tout", command = self.apply_all).grid(row = 0, column = 6, padx = 10)

    def update(self):
        # Capture de la trame vidéo
        ret, frame = self.vid.read()

        # Application des traitements choisis
        if self.apply_sepia:
            frame = self.apply_sepia_filter(frame)
        if self.overlay_snowflakes:
            frame = self.overlay_snowflakes_effect(frame)
        if self.change_background:
            frame = self.change_background_function(frame, cv2.imread("images/fond-563x612.jpg"))
        if self.detect_eye_flag:
            frame = self.detect_eye_function(frame)
            
        if self.detect_nose_flag:
            frame = self.detect_nose_function(frame)
            
        if self.detect_barbe_flag:
            frame = self.detect_barbe_function(frame)

        # Mise à jour du canva avec la nouvelle image
        self.photo = ImageTk.PhotoImage(image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        self.canvas.create_image(0, 0, image = self.photo, anchor = tk.NW)

        # Appel récursif périodique
        self.window.after(10, self.update)

    def apply_sepia_filter(self, frame):
        sepia_matrix = np.array([[0.393, 0.769, 0.189],
                                [0.349, 0.686, 0.168],
                                [0.272, 0.534, 0.131]])

        sepia_frame = cv2.transform(frame, sepia_matrix)
        sepia_frame = np.clip(sepia_frame, 0, 255).astype(np.uint8)
        
        return sepia_frame

    def overlay_snowflakes_effect(self, frame):
        num_snowflakes = 50
        for _ in range(num_snowflakes):
            x, y = np.random.randint(0, frame.shape[1]), np.random.randint(0, frame.shape[0])
            frame[y:y + 4, x:x + 4] = [255, 255, 255]

        return frame

    def change_background_function(self, sujet, fond):
        height, width, _ = sujet.shape
        fond = cv2.resize(fond, (width, height))

        '''
        face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')
        gray = cv2.cvtColor(sujet, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        result = sujet.copy()
    
        for i in range(faces.shape[0]):
            center = (faces[i, 0] + int(faces[i, 2] * 0.5), faces[i, 1] + int(faces[i, 3] * 0.5))
            axes = (int(faces[i, 2] * 0.5), int(faces[i, 3] * 0.5))

            mask_ellipse = np.zeros_like(gray)

            # Dessiner l'ellipse blanche sur un masque
            cv2.ellipse(mask_ellipse, center, axes, 0, 0, 360, (255), -1)

            distance_threshold = 2 * max(axes)
            dist_transform = cv2.distanceTransform(mask_ellipse, cv2.DIST_L2, 5)
            print(distance_threshold)
            mask_distance = np.uint8(dist_transform <= distance_threshold)
        '''

        lower_white = np.array([160, 160, 160])
        upper_white = np.array([255, 255, 255])
        mask_white = cv2.inRange(sujet, lower_white, upper_white)

        '''
        final_mask = cv2.bitwise_and(mask_distance, mask_white)

        mask_inv = cv2.bitwise_not(final_mask)

        result = cv2.bitwise_and(result, result, mask=mask_inv)
        result += cv2.bitwise_and(fond, fond, mask=final_mask)
        '''

        mask_inv = cv2.bitwise_not(mask_white)

        result = cv2.bitwise_and(sujet, sujet, mask=mask_inv)
        result += cv2.bitwise_and(fond, fond, mask=mask_white)

        return result
    
    def detect_eye_function(self, image):
        # Conversion en niveaux de gris
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Utilisez un classificateur Haar pour détecter la bouche
        eye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye_tree_eyeglasses.xml')
       
        # Détecter les yeux dans l'image
        eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Chargement de l'image par défaut pour les lunettes
        default_glasses_image = cv2.imread('images/lunette.png', cv2.IMREAD_UNCHANGED)
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
        default_nose_image = cv2.imread('images/nezchien.png', cv2.IMREAD_UNCHANGED)

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