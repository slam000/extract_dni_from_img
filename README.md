# Extracción de Texto de Imágenes y PDFs
Este proyecto es un script de Python que se utiliza para extraer texto de imágenes y documentos PDF, y luego buscar números de documentos (DNI, NIE, pasaporte) en el texto extraído.

## Dependencias
El proyecto depende de las siguientes bibliotecas de Python:

OpenCV (cv2)
Matplotlib
NumPy
PyTesseract
os
re
pdf2image

## Funciones
El script consta de varias funciones:

**ocr(image)**: Esta función toma una imagen como entrada y devuelve el texto extraído de la imagen.

**convertir_a_imagen(pdf)**: Esta función toma un archivo PDF como entrada y convierte la primera página en una imagen JPEG.

**extraerTexto(img_color)**: Esta función toma una imagen en color como entrada y la convierte en escala de grises. Luego binariza la imagen, realiza una operación de apertura para eliminar el ruido, invierte la imagen, la amplía y extrae el texto.

**buscar_dni(texto_completo)**: Esta función busca diferentes formatos de números de documentos en el texto extraído.

**extrae_paths_imagenes(path)**: Esta función busca todos los archivos en un directorio dado. Si el archivo es un PDF, lo convierte en una imagen y agrega la ruta del archivo a una lista. Si el archivo es una imagen, simplemente agrega la ruta del archivo a la lista.

**main()**: Esta es la función principal que llama a las otras funciones y ejecuta el script.

## Uso
Para usar este script, simplemente ejecute el archivo app.py en su terminal con Python. Asegúrese de tener todas las dependencias instaladas y de tener algunas imágenes o archivos PDF en el directorio 'imagenes_prueba' para que el script pueda procesarlos.

## Contribuciones
Las contribuciones a este proyecto son bienvenidas. Si encuentra un error o tiene una sugerencia para una mejora, no dude en abrir un problema o enviar una solicitud de extracción.

## Licencia
Este proyecto está licenciado bajo los términos de la licencia MIT.
