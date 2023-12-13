import cv2
from matplotlib import pyplot as plt
import numpy as np
import pytesseract
from pytesseract import Output
import os
import re
from pdf2image import convert_from_path
import pandas as pd
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ResourceNotFoundError





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


# Carga CONTRATOS_DNIS.csv y devuelve el contrato_id en base al dni
def dame_contrato_id(dni):
    """
    Carga CONTRATOS_DNIS.csv y devuelve el contrato_id en base al dni

    Args:
        dmi (str): DNI, NIE o pasaporte.

    Returns:
        str: Contrato_id.
    
    Example csv file:
        sociedad;contrato;interl_comercial;dni;contrato_id
        2000;2300000000000;1000000003;12345678A;2000/2300000000000
    """
    # Cargar el archivo csv
    df = pd.read_csv('check-contract/CONTRATOS_DNIS.csv', sep=';')
    # Buscar el contrato_id en base al dni, si no lo encuentra devuelve None
    contrato_id = None
    if dni in df['dni'].values:
        contrato_id = df[df['dni'] == dni]['contrato_id'].values[0]
    return contrato_id


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
    # Invetir la imagen
    invert_image = 255 - opening_image
    # ampliar la imagen
    invert_image = cv2.resize(invert_image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    # Guardar la imagen ampliada
    cv2.imwrite('/imagenes/temp_.jpg', invert_image)
    
    
    img = img_color
    # Convertir a escala de grises
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    thresh = cv2.resize(thresh, None, fx=1.25, fy=1.25)
    blur = cv2.GaussianBlur(thresh, (5,5), 0)
    detect_text = pytesseract.image_to_string(blur, lang='spa', config='--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNÑOPQRSTUVWXYZ')
    print(f'Texto detectado con nuevo método: {detect_text}')
    
    # Extraer el texto de la imagen
    data_image= pytesseract.image_to_data(invert_image, output_type=Output.DICT)
    
    texto_completo = ' '.join(data_image['text'])
    # Eliminar los caracteres especiales
    texto_completo = re.sub(r'[^A-Za-z0-9]+', ' ', texto_completo)
    
    # Convertir texto a mayúsculas
    texto_completo = detect_text.upper()
        
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
    print(f'Texto completo: {texto_completo}')
    
    # dni = re.search(r'\b\d{8}[A-Z]\b', texto_completo)
    dni = re.search(r'\d{8}[A-Za-z]', texto_completo)
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
                return 'None'
    
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

# extraer paths de las imagenes desde el csv
def extrae_paths_imagenes_csv(path_csv):
    csv_file = pd.read_csv(path_csv, sep=';')
    # Extraer de la columna path y container_name
    path_imagenes = []
    for index, row in csv_file.iterrows():
        # añado el path de la imagen y el container_name
        path_imagenes.append((row['path'], row['container_name']))
        
    return path_imagenes    

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
    # paths_imagenes = extrae_paths_imagenes('imagenes_prueba')
    csv_file = 'test.csv'
    paths_imagenes = extrae_paths_imagenes_csv(csv_file)
    
    # Crear un diccionario para almacenar los resultados de la extracción de DNIs, NIEs, contenedores y paths
    resultados = {'DNI': [], 'Contrato': [], 'container_name': [], 'path': []}

    for path in paths_imagenes:
        container_name = path[1]
        blob_name = path[0]
                
        print(f'Procesando imagen: {blob_name} correspondiente al container: {container_name}')
        print('-----------------')
        print('Conectando con blob storage...')
        
        blob_service_client = BlobServiceClient(account_url="https://maccstoragerentaldocpro.blob.core.windows.net", credential="xPTRKus+jDQ4d4CatSG4A1k/+kj+q06x5XAUwGYqa0VrYL/ZsSQ6kZCj+SvAlX7DbYORhp5egT/L+ASt3Dk5tw==")
        blob_client = blob_service_client.get_blob_client(container_name, blob_name)
        
        print('Descargando imagen...')
        
        
        # Descargar la imagen del blob storage al directorio local imagenes
        try:
            with open(blob_name, "wb") as my_blob:
                blob_data = blob_client.download_blob()
                blob_data.readinto(my_blob)

            # Cargar la imagen
            img_color = cv2.imread(blob_name)
            
            # Si el path es una imagen jpg, png o jpeg
            if blob_name.endswith('.jpg') or blob_name.endswith('.png') or blob_name.endswith('.jpeg'):            
                # Identificar frente o dorso
                print('Identificando frente o dorso...')
                
                fd = identificar_frente_dorso(img_color)
                if fd == 'Frente':
                    texto_completo = extraerTexto(img_color)
                    dni = buscar_dni(texto_completo)
                    print(f'DNI: {dni}')
                    
                    contrato = dame_contrato_id(dni)
                    
                    # Añadir los resultados al diccionario
                    resultados['DNI'].append(dni)
                    resultados['Contrato'].append(contrato)
                    resultados['container_name'].append(container_name)
                    resultados['path'].append(blob_name)
                    
                    
                    print(f'Contrato: {contrato}')
                    
                else:
                    print('Dorso')
                    texto_completo = extraerTexto(img_color)
                    # La expresión regular para un DNI es \d{8}[A-Za-z]< 
                    dni = re.findall(r'\b\d{8}[A-Z]\b', texto_completo)
                    if dni:
                        contrato = dame_contrato_id(dni)
                        # Añadir los resultados al diccionario
                        resultados['DNI'].append(dni)
                        resultados['Contrato'].append(contrato)
                        resultados['container_name'].append(container_name)
                        resultados['path'].append(blob_name)
                        
                        print(f'Texto completo: {f'Texto extraido del dorso de la imagen {texto_completo}. DNI: {dni}'}')
                    else:
                        # Añadir los resultados al diccionario
                        resultados['DNI'].append('dorso')
                        resultados['Contrato'].append('dorso')
                        resultados['container_name'].append(container_name)
                        resultados['path'].append(blob_name)
                        print('No se encontró número documento.')
                        
                print('-----------------')
                print('-----------------')
            
            # Si el path es un pdf
            elif blob_name.endswith('.pdf'):
                # Convertir el pdf a imagen
                print(f'El blob {blob_name} es un pdf.')
            
            print(f'eliminando imagen: {blob_name} en entorno local.')
            # Eliminar la imagen local
            os.remove(blob_name)
            
            print('-----------------')
            print('-----------------')
            
    
        except ResourceNotFoundError:
            print(f'No se encontró el blob {blob_name} en el container {container_name}')
            print('-----------------')
            print('-----------------')
            continue
    
    # Imprimir los resultados por pantalla y guardarlos en un csv
    df = pd.DataFrame(resultados)
    print(df)
    df.to_csv('resultados.csv', index=False)
    print('-----------------')
    
        
    
        
if __name__ == '__main__':
    main()

