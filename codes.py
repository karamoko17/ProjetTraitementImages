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
        self.detect_mouth = False
        
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

    def detect_mouth(self):
        self.detect_mouth = not self.detect_mouth

    def apply_all(self):
        self.apply_sepia = True
        self.overlay_snowflakes = True
        self.change_background = True
        # self.detect_mouth = True

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
        tk.Button(menu_frame, text = "Barbe", command = self.detect_mouth).grid(row = 0, column = 3, padx = 10)
        tk.Button(menu_frame, text = "Chien", command = self.apply_all).grid(row = 0, column = 4, padx = 10)
        tk.Button(menu_frame, text = "Lunettes", command = self.apply_all).grid(row = 0, column = 5, padx = 10)
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
        if self.detect_mouth:
            frame = self.detect_mouth_function(frame)

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
    
    def detect_mouth_function(self, image):
        # Conversion en niveaux de gris
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Utilisez un classificateur Haar pour détecter la bouche
        mouth_cascade = cv2.CascadeClassifier('D:/M1 Lyon 2/Traitement d\'image/ProjetTraitementImages/haarcascades/haarcascades/haarcascade_mouth.xml')
        mouths = mouth_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
         
        # Chargement de l'image par défaut pour la bouche
        self.default_mouth_image = cv2.imread('D:/M1 Lyon 2/Traitement d\'image/ProjetTraitementImages/images/barbe.png', cv2.IMREAD_UNCHANGED)

        # Dessiner l'image par défaut de la bouche dans les zones détectées
        for (x, y, w, h) in mouths:
            # Redimensionner l'image de la bouche pour s'adapter à la zone détectée
            resized_mouth_image = cv2.resize(self.default_mouth_image, (w, h))

            # Obtenir les indices des pixels non nuls dans le canal alpha
            alpha_indices = resized_mouth_image[:, :, 3] > 0

            # Superposer l'image de la bouche sur l'image principale
            image[y:y+h, x:x+w][alpha_indices] = (
                resized_mouth_image[:, :, :3][alpha_indices] * (resized_mouth_image[:, :, 3][alpha_indices] / 255.0) +
                image[y:y+h, x:x+w][alpha_indices] * (1.0 - resized_mouth_image[:, :, 3][alpha_indices] / 255.0)
            )

        return image
    
    def on_close(self):
        if self.vid.isOpened():
            self.vid.release()
        self.window.destroy()


# Création de l'application
root = tk.Tk()
app = App(root, "App")
root.mainloop()