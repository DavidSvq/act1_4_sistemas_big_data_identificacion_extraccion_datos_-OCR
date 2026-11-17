# IMPORTACIÓN DE LIBRERÍAS
import pytesseract
from pathlib import Path
import csv
import cv2
import numpy as np

# PASO 1: CARGAR LA IMAGEN ADAPTATIVA

imagen = cv2.imread(
    "resultados/imagen_ampliada.png",
    cv2.IMREAD_GRAYSCALE
)

if imagen is None:
    raise FileNotFoundError(
        "No se ha encontrado la imagen adaptativa."
    )

# Carpetas de trabajo
carpeta_textos = Path("resultados/ocr")
carpeta_textos.mkdir(parents=True, exist_ok=True)

# PASO 2: DETECCIÓN DE LA ESTRUCTURA DE LA TABLA

def detectar_lineas_tabla(imagen):

    alto, ancho = imagen.shape[:2]

    # Convertir imagen a binario
    _, binaria = cv2.threshold(
        imagen,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Detectar líneas horizontales
    kernel_horizontal = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(20, ancho // 30), 1)
    )

    lineas_horizontales = cv2.morphologyEx(
        binaria,
        cv2.MORPH_OPEN,
        kernel_horizontal
    )

    # Detectar líneas verticales
    kernel_vertical = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (1, max(10, alto // 30))
    )

    lineas_verticales = cv2.morphologyEx(
        binaria,
        cv2.MORPH_OPEN,
        kernel_vertical
    )

    # Obtener proyecciones de las líneas
    proyeccion_horizontal = np.sum(
        lineas_horizontales > 0,
        axis=1
    )

    proyeccion_vertical = np.sum(
        lineas_verticales > 0,
        axis=0
    )

    # Localizar coordenadas de las líneas
    posiciones_h = np.where(
        proyeccion_horizontal > ancho * 0.5
    )[0]

    posiciones_v = np.where(
        proyeccion_vertical > alto * 0.5
    )[0]

    # Agrupar coordenadas consecutivas
    def agrupar_lineas(posiciones):

        grupos = []

        if len(posiciones) == 0:
            return grupos

        inicio = posiciones[0]
        anterior = posiciones[0]

        for posicion in posiciones[1:]:

            if posicion - anterior > 3:
                grupos.append((inicio + anterior) // 2)
                inicio = posicion

            anterior = posicion

        grupos.append((inicio + anterior) // 2)

        return grupos

    filas = agrupar_lineas(posiciones_h)
    columnas = agrupar_lineas(posiciones_v)

    # Comprobar detección
    if len(filas) < 3 or len(columnas) < 4:
        raise ValueError(
            "No se han detectado correctamente las líneas de la tabla."
        )

    return filas, columnas


filas, columnas = detectar_lineas_tabla(imagen)

# PASO 3: RECORTAR LAS CELDAS Y EXTRAER EL TEXTO

def extraer_celda(imagen, x1, y1, x2, y2):

    # Recortar la celda evitando incluir los bordes
    margen = 3

    celda = imagen[
        y1 + margen:y2 - margen,
        x1 + margen:x2 - margen
    ]

    # Reconocimiento OCR con PSM por defecto
    texto = pytesseract.image_to_string(
        celda,
        lang="eng"
    )

    return texto.replace("\n", " ").replace("\r", " ").strip()

# PASO 4: EXTRAER LAS TRES COLUMNAS

datos_tabla = []

# Recorrer todas las filas, incluido el encabezado
for i in range(len(filas) - 1):

    y1 = filas[i]
    y2 = filas[i + 1]

    # Extraer las tres columnas
    metodo = extraer_celda(
        imagen,
        columnas[0],
        y1,
        columnas[1],
        y2
    )

    endpoint = extraer_celda(
        imagen,
        columnas[1],
        y1,
        columnas[2],
        y2
    )

    proposito = extraer_celda(
        imagen,
        columnas[2],
        y1,
        columnas[3],
        y2
    )

    datos_tabla.append([
        metodo,
        endpoint,
        proposito
    ])

# PASO 5: GENERAR EL CSV FINAL

archivo_csv = carpeta_textos / "imagen_ampliada.csv"

with open(
    archivo_csv,
    "w",
    newline="",
    encoding="utf-8-sig"
) as archivo:

    escritor = csv.writer(
        archivo,
        delimiter=";"
    )

    escritor.writerows(datos_tabla)

print(f"CSV generado correctamente: {archivo_csv}")

