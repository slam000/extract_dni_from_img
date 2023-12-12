import cv2
from matplotlib import pyplot as plt
import numpy as np
import pytesseract
from pytesseract import Output
import os
import re
from pdf2image import convert_from_path

def ocr(image):
    custom_config = r'-l spa - psm 11'
    text = pytesseract.image_to_string(image,config=custom_config )
    return text

def convertir_a_imagen(pdf):
    # Convertir el pdf a imagen solo la primera página
    img_color = convert_from_path(pdf)[0]
    # Guardar la imagen
    img_color.save('temp.jpg', 'JPEG')
    # Cargar la imagen
    img_color = cv2.imread('temp.jpg')
    # Eliminar la imagen temporal
    os.remove('temp.jpg')
    return img_color

def extraerTexto(img_color):
    # Transformar a escala de grises
    img_gris = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    # Binarizar la imagen
    thresh_img = cv2.threshold(img_gris, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,2))
    opening_image = cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN, kernel,iterations=1)

    # Mostrar la imagen
    #plt.imshow(opening_image, cmap='gray')
    #plt.show()


    # Invetir la imagen
    invert_image = 255 - opening_image
    # ampliar la imagen
    invert_image = cv2.resize(invert_image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    data_image= pytesseract.image_to_data(invert_image, output_type=Output.DICT)


    texto_completo = ' '.join(data_image['text'])
    return texto_completo

def buscar_dni(texto_completo):
    dni = re.search(r'\b\d{8}[A-Z]\b', texto_completo)
    if dni:
        print(f"DNI encontrado: {dni.group()}")
        return dni.group()
    
    else:
        # Expresion regunlar para encontrar el NIE (X, Y, Z + 7 digitos + 1 letra)
        nie = re.search(r'\b[X-Z]\d{7}[A-Z]\b', texto_completo)
        # Imprimir el NIE, si se encontró
        if nie:
            print(f"NIE encontrado: {nie.group()}")
            return nie.group()
        
        else:
            # Expresion regunlar para encontrar el pasaporte (3 letras + 6 digitos)
            pasaporte = re.search(r'\b[A-Z]{3}\d{6}\b', texto_completo)
            # Imprimir el pasaporte, si se encontró
            if pasaporte:
                print(f"Pasaporte encontrado: {pasaporte.group()}")
                return pasaporte.group()
            
            else:
                print("No se encontró número documento.")
                return "DNI no encontrado"
    
def extrae_paths_imagenes(path):
    paths_imagenes = []
    # Extraer los paths de las imagenes
    for file in os.listdir(path):
        if file.endswith('.pdf'):
            path_pdf = os.path.join(path, file)
            img_color = convertir_a_imagen(path_pdf)
            paths_imagenes.append(path_pdf)
        else:
            path_imagen = os.path.join(path, file)
            paths_imagenes.append(path_imagen)
    
    return paths_imagenes

# identificar el tipo de documento (DNI, NIE, Pasaporte)
def identificar_documento(texto_completo):
    # Idemtificar si es DNI o NIE por medop de expresiones regulares
    if re.search(r'\b\d{8}[A-Z]\b', texto_completo):
        print('DNI')
        return 'DNI'
    elif re.search(r'\b[X-Z]\d{7}[A-Z]\b', texto_completo):
        print('NIE')
        return 'NIE'
    elif re.search(r'\b[A-Z]{3}\d{6}\b', texto_completo):
        print('Pasaporte')
        return 'Pasaporte'
    else:
        print('No se encontró número documento')
        
    
    pass

# Procesar texto dorso del dni
def procesar_dorso_dni(texto_completo):
    numero = re.search(r'\b\d{8}[A-Z]\b', texto_completo)
    if numero:
        print(f"Número encontrado: {numero.group()}")
        
    else:
        print("No se encontró número documento.")
        

# identificar frente o dorso del dni detectando si contiene foto con cara
def identificar_frente_dorso(imagen):
    # Transformar a escala de grises
    img_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    # Comprobar si la imagen tiene cara
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(img_gris, 1.3, 5)
    if len(faces) > 0:
        print('Frente')
        return 'Frente'
    else:
        print('Dorso')
        return 'Dorso'


# Función principal
def main():
    paths_imagenes = []
    paths_imagenes = extrae_paths_imagenes('imagenes_prueba')
    
    print(paths_imagenes)
    
    for path in paths_imagenes:

        # Cargar la imagen
        img_color = cv2.imread(path)
        
        # Identificar frente o dorso
        fd = identificar_frente_dorso(img_color)
        
        if fd == 'Frente':
            texto_completo = extraerTexto(img_color)
            dni = buscar_dni(texto_completo)
            print(dni)
        else:
            print('Dorso')
            texto_completo = extraerTexto(img_color)
            print(texto_completo)
            procesar_dorso_dni(texto_completo)
            
        print('-----------------')
        print('-----------------')
        
        
        
if __name__ == '__main__':
    main()

