# image.py
import io
import tkinter as tk
from tkinter import filedialog
from PIL import Image # type: ignore
import numpy as np

def resize(image_bytes, width, height):
    """Recibe los bytes de una imagen, la redimensiona y la devuelve como un array de NumPy."""
    # Cargamos la imagen desde los bytes en memoria
    image = Image.open(io.BytesIO(image_bytes))
    
    # Nos aseguramos de que esté en modo RGB (3 canales: R, G, B)
    if image.mode != 'RGB':
        image = image.convert('RGB')
        
    image = image.resize((width, height), Image.Resampling.LANCZOS)
    
    # Convertimos la imagen de PIL directamente a un array numérico de NumPy
    return np.array(image)

def get_image_file(filepath):
    """Lee un archivo del disco de forma segura y devuelve sus bytes."""
    with open(filepath, 'rb') as file:
        return file.read()

def get_folder_path():
    """Abre un cuadro de diálogo para seleccionar una carpeta."""
    root = tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory(title="Select Dataset Folder")
    return folder_path if folder_path else None

def get_file():
    """Abre un cuadro de diálogo para seleccionar un archivo específico (para cargar el modelo)."""
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select Model File",
        filetypes=[("Keras Models", "*.keras"),
                   ("Image Files", "*.png *.jpg *.jpeg *.bmp *.webp"),
                   ("All files", "*.*")]
    )
    return file_path if file_path else None