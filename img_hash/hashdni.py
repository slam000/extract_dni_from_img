import cv2
from PIL import Image
import imagehash
import json
import os

def detectar_bordes(imagen_path):
    imagen = cv2.imread(imagen_path, cv2.IMREAD_GRAYSCALE)
    bordes = cv2.Canny(imagen, threshold1=100, threshold2=200)
    return bordes

def encontrar_contorno_dni(bordes):
    contornos, _ = cv2.findContours(bordes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dni_contorno = max(contornos, key=cv2.contourArea)
    return dni_contorno

def extraer_roi(imagen_path, contorno):
    imagen = cv2.imread(imagen_path)
    x, y, w, h = cv2.boundingRect(contorno)
    roi = imagen[y:y+h, x:x+w]
    return roi


def calcular_hash(roi, size=(128, 128)):
    imagen = Image.fromarray(roi)  # Convertir de OpenCV a PIL
    imagen = imagen.convert("L")
    imagen = imagen.resize(size, Image.LANCZOS)
    hash = imagehash.average_hash(imagen)
    return hash


def es_imagen_similar(hash_nueva_imagen, hashes_almacenados, umbral=25):
    for hash_almacenado in hashes_almacenados.values():
        if abs(hash_nueva_imagen - imagehash.hex_to_hash(hash_almacenado)) <= umbral:
            return True
    return False

# Recorrer el directorio de imagenes_front y crear un diccionario con sus hashes
hashes = {}
for imagen_path in os.listdir("img_hash/imagenes_front"):
    path_completo = "img_hash/imagenes_front/" + imagen_path
    bordes = detectar_bordes(path_completo)
    contorno = encontrar_contorno_dni(bordes)
    roi = extraer_roi(path_completo, contorno)
    
    # Mostrar ROI
    # s
    cv2.namedWindow("ROI", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("ROI", 600, 600)
    cv2.imshow("ROI", roi)

    
    hashes[imagen_path] = str(calcular_hash(roi))

    print(imagen_path, hashes[imagen_path])

# Calcular hash de una imagen de prueba
path_prueba = "imagenes/test2.jpg"
bordes_prueba = detectar_bordes(path_prueba)
contorno_prueba = encontrar_contorno_dni(bordes_prueba)
roi_prueba = extraer_roi(path_prueba, contorno_prueba)
hash_img = calcular_hash(roi_prueba)

similitud_encontrada = es_imagen_similar(hash_img, hashes)
print("¿Es similar a alguna imagen almacenada?", similitud_encontrada)

