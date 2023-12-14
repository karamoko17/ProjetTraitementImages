import tkinter as tk
from PIL import Image, ImageTk

def action():
    print("Bouton rond cliqué")

root = tk.Tk()

# Liste pour stocker les images
images = []

def create_image_button(frame, image_path, width, height, action):
    img = Image.open(image_path)
    img.thumbnail((width, height))
    photo = ImageTk.PhotoImage(img)
    images.append(photo)
    btn = tk.Button(frame, image=photo, command=action, bd=0, relief=tk.FLAT)
    btn.pack(padx=10, pady=10)
    return btn

btn_width = 50
btn_height = 50

create_image_button(root, "images/beach.png", btn_width, btn_height, action)
create_image_button(root, "images/sepia.png", btn_width, btn_height, action)
create_image_button(root, "images/snowflakes.png", btn_width, btn_height, action)
create_image_button(root, "images/barbe.png", btn_width, btn_height, action)
create_image_button(root, "images/chien.png", btn_width, btn_height, action)
create_image_button(root, "images/lunettes.png", btn_width, btn_height, action)
create_image_button(root, "images/all.png", btn_width, btn_height, action)

root.mainloop()




'''
# Fonction pour créer un bouton rond avec une image redimensionnée tout en conservant la proportion
def creer_bouton_rond(image_path, width, height):
    image = Image.open(image_path)
    image.thumbnail((width, height))  # Redimensionner l'image tout en conservant la proportion
    photo = ImageTk.PhotoImage(image)
    bouton = tk.Button(root, image=photo, command=action, bd=0, relief=tk.FLAT)
    bouton.image = photo  # Garder une référence à l'image pour éviter la suppression par le garbage collector
    bouton.pack(padx=10, pady=10)
    return bouton

# Taille souhaitée pour les images des boutons
largeur_bouton = 100
hauteur_bouton = 100

# Création de boutons ronds avec des images redimensionnées tout en conservant la proportion
bouton_rond_1 = creer_bouton_rond("images/barbe.png", largeur_bouton, hauteur_bouton)
bouton_rond_2 = creer_bouton_rond("images/lunette.png", largeur_bouton, hauteur_bouton)
bouton_rond_3 = creer_bouton_rond("images/nezchien.png", largeur_bouton, hauteur_bouton)

root.mainloop()
'''

