import cv2
from matplotlib import pyplot as plt
import numpy as np
import pytesseract
from pytesseract import Output
import os
import re
from pdf2image import convert_from_path

def ocr(image):
    """
    Realiza la extracción de texto de una imagen utilizando OCR.

    Parámetros:
    - image: La imagen de entrada para realizar la extracción de texto.

    Retorna:
    - El texto extraído de la imagen.
    """
    custom_config = r'-l spa - psm 11'
    text = pytesseract.image_to_string(image,config=custom_config )
    return text

def convertir_a_imagen(pdf):
    """
    Convierte un archivo PDF en una imagen.

    Args:
        pdf (str): Ruta del archivo PDF.

    Returns:
        numpy.ndarray: Imagen convertida.
    """
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
    """
    Extrae el texto de una imagen en color.

    Parameters:
    img_color (numpy.ndarray): La imagen en color de la cual se desea extraer el texto.

    Returns:
    str: El texto extraído de la imagen.
    """
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
    """
    Busca y devuelve el número de documento (DNI, NIE o pasaporte) en un texto dado.

    Parámetros:
    texto_completo (str): El texto en el que se buscará el número de documento.

    Retorna:
    str: El número de documento encontrado (DNI, NIE o pasaporte) o un mensaje de error si no se encontró.

    Ejemplo:
    >>> buscar_dni("Mi DNI es 12345678A")
    DNI encontrado: 12345678A
    '12345678A'
    """
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
    """
    Extrae los paths de las imágenes en el directorio especificado.

    Args:
        path (str): Ruta del directorio.

    Returns:
        list: Lista de paths de las imágenes.
    """
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
    """
    Identifica el tipo de documento a partir de un texto completo.

    Args:
        texto_completo (str): El texto completo que contiene el número de documento.

    Returns:
        str: El tipo de documento identificado ('DNI', 'NIE', 'Pasaporte') o 'No se encontró número documento' si no se encuentra ningún número de documento.
    """
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
        return 'No se encontró número documento'

# Procesar texto dorso del dni
def procesar_dorso_dni(texto_completo):
    """
    Procesa el texto completo del dorso del DNI y busca el número de documento.

    Args:
        texto_completo (str): El texto completo del dorso del DNI.

    Returns:
        None
    """
    numero = re.search(r'\b\d{8}[A-Z]\b', texto_completo)
    if numero:
        print(f"Número encontrado: {numero.group()}")
    else:
        print("No se encontró número documento.")
        

# identificar frente o dorso del dni detectando si contiene foto con cara
def identificar_frente_dorso(imagen):
    """
    Identifica si una imagen corresponde al frente o al dorso de un documento.
    
    Parámetros:
    - imagen: La imagen a analizar.
    
    Retorna:
    - 'Frente' si la imagen corresponde al frente de un documento.
    - 'Dorso' si la imagen corresponde al dorso de un documento.
    """
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
    """
    Función principal que procesa las imágenes de prueba.
    """
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

