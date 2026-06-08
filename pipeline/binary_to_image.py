import numpy as np #trasformo byte in array numerico
from PIL import Image #Pillow, per manipolare immagine
import math
import os

def binary_to_image(input_path, output_path):
    #gestione file
    try:
        with open(input_path, "rb") as fb: #più sicura, così si chiude anche se c'è errore
            byte_data = fb.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"File non trovato: {input_path}")
    
    #metto byte in array
    pixel_array = np.frombuffer(byte_data, np.uint8)
    
    #dimensioni immagine + padding
    size = math.ceil(math.sqrt(len(pixel_array))) #per arrotondare per eccesso
    padding = (size*size) - len(pixel_array)
    pixel_array = np.append(pixel_array, np.zeros(padding, dtype=np.uint8))
    
    #reshape da lineare a 2D
    pixel_array = pixel_array.reshape((size, size))
    
    #crea immagine
    img = Image.fromarray(pixel_array)
    
    #resize
    img = img.resize((224, 224))
    
    #save
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    img.save(output_path)
    
    try:
        test = Image.open(output_path)
        print("Formato:", test.format)
        print("Dimensioni:", test.size)
        test.close()
    except Exception as e:
        print("Errore dopo il salvataggio:", e)
    
