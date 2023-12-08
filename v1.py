import cv2
from PIL import Image, ImageTk, ImageFilter
import tkinter as tk
from tkinter import filedialog
import numpy as np
import threading

class WebcamApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        # Initialisation de la capture vidéo à partir de la webcam
        self.video_source = 0  # Utilisez 0 pour la webcam par défaut
        self.vid = cv2.VideoCapture(self.video_source)

        # Création d'un canevas pour afficher la vidéo
        self.canvas = tk.Canvas(window, width=self.vid.get(cv2.CAP_PROP_FRAME_WIDTH), height=self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.canvas.pack()

        # Création d'un menu interactif
        self.create_menu()

        # Drapeaux pour indiquer quels traitements doivent être appliqués
        self.apply_sepia_flag = False
        self.overlay_image_flag = False
        self.overlay_snowflakes_flag = False
        self.change_background_flag = False
        
        # Lancement du processus de mise à jour de la vidéo
        self.update()

        # Fermeture de la fenêtre
        self.window.protocol("WM_DELETE_WINDOW", self.on_close)
        
    def create_menu(self):
        menu_frame = tk.Frame(self.window)
        menu_frame.pack(pady=10)

        # Bouton pour changer le fond
        tk.Button(menu_frame, text="Changer le fond", command=self.change_background).grid(row=0, column=0, padx=10)

        # Bouton pour appliquer un filtre sépia
        tk.Button(menu_frame, text="Filtre Sépia", command=self.apply_sepia).grid(row=0, column=1, padx=10)

        # Bouton pour incrustrer une image
        tk.Button(menu_frame, text="Incruster Image", command=self.overlay_image).grid(row=0, column=2, padx=10)

        # Bouton pour incrustrer des flocons de neige
        tk.Button(menu_frame, text="Flocons de neige", command=self.overlay_snowflakes).grid(row=0, column=3, padx=10)

        # Bouton pour lancer tous les traitements en même temps
        tk.Button(menu_frame, text="Appliquer Tout", command=self.apply_all).grid(row=0, column=4, padx=10)

    def apply_sepia(self):
        self.apply_sepia_flag = not self.apply_sepia_flag

    def overlay_image(self):
        self.overlay_image_flag = not self.overlay_image_flag
        if self.overlay_image_flag:
            # Demander à l'utilisateur de choisir une image à incrustrer
            file_path = filedialog.askopenfilename()
            if file_path:
                self.overlay_image_path = file_path

    def overlay_snowflakes(self):
        self.overlay_snowflakes_flag = not self.overlay_snowflakes_flag

    def change_background(self):
        self.change_background_flag = not self.change_background_flag

    def apply_all(self):
        self.apply_sepia_flag = True
        self.overlay_image_flag = True
        self.overlay_snowflakes_flag = True
        self.change_background_flag = True

    def update(self):
        # Capture la trame vidéo
        ret, frame = self.vid.read()

        # Applique les traitements choisis
        if self.apply_sepia_flag:
            frame = self.apply_sepia_filter(frame)

        if self.overlay_image_flag and hasattr(self, 'overlay_image_path'):
            overlay_image = cv2.imread(self.overlay_image_path, cv2.IMREAD_UNCHANGED)
            frame = self.overlay_image_on_face(frame, overlay_image)
            
        if self.overlay_snowflakes_flag:
            frame = self.overlay_snowflakes_effect(frame)

        if self.change_background_flag:
            frame = self.change_background_function(frame)

        # Met à jour le canevas avec la nouvelle image
        self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        # Mise à jour périodique
        self.window.after(10, self.update)

    def apply_sepia_filter(self, frame):
       # Applique un filtre sépia à l'image entière avec OpenCV
        sepia_matrix = np.array([[0.393, 0.769, 0.189],
                                [0.349, 0.686, 0.168],
                                [0.272, 0.534, 0.131]])

        sepia_frame = cv2.transform(frame, sepia_matrix)
        sepia_frame = np.clip(sepia_frame, 0, 255).astype(np.uint8)
        
        return sepia_frame


    def overlay_image_on_face(self, frame, overlay_image):
        # Incruste une image sur le visage
        # Implémentez la logique pour détecter le visage et ajuster les coordonnées de l'overlay en conséquence
        # Dans cet exemple, l'image d'overlay est simplement redimensionnée et superposée au coin supérieur gauche de l'image
        h, w, _ = frame.shape
        overlay_image = cv2.resize(overlay_image, (w, h))
        alpha_overlay = overlay_image[:, :, 3] / 255.0
        alpha_frame = 1.0 - alpha_overlay

        for c in range(0, 3):
            frame[:, :, c] = (alpha_overlay * overlay_image[:, :, c] +
                              alpha_frame * frame[:, :, c])

        return frame

    def overlay_snowflakes_effect(self, frame):
        # Incruste des flocons de neige animés dans le fond
        # Vous pouvez implémenter cette fonctionnalité en ajoutant des éléments interactifs (par exemple, des flocons de neige) dans le fond
        # Ici, nous allons simplement superposer des pixels blancs à des positions aléatoires
        num_snowflakes = 50
        for _ in range(num_snowflakes):
            x, y = np.random.randint(0, frame.shape[1]), np.random.randint(0, frame.shape[0])
            frame[y:y + 4, x:x + 4] = [255, 255, 255]  # Les flocons de neige sont représentés par des pixels blancs

        return frame

    def change_background_function(self, frame):
        # Rend le fond de l'image en blanc et conserve l'utilisateur en couleur
        # Vous pouvez ajuster cette fonction selon vos besoins spécifiques
        mask = np.ones_like(frame) * 255  # Crée un masque blanc de la même taille que l'image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Utilisez un algorithme de détection de visage pour trouver les coordonnées du visage
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # Remplace le fond par blanc dans la région du visage
            mask[y:y+h, x:x+w, :] = frame[y:y+h, x:x+w]

        return mask
    
    
    def on_close(self):
        # Libération de la capture vidéo lors de la fermeture de l'application
        if self.vid.isOpened():
            self.vid.release()
        self.window.destroy()

# Création de l'application
root = tk.Tk()
app = WebcamApp(root, "Webcam App")
root.mainloop()
