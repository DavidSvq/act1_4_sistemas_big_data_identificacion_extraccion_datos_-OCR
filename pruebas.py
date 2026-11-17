# IMPORTACIÓN DE LIBRERÍAS

import pytesseract
from pathlib import Path
import csv

import cv2
from matplotlib import pyplot as plt
import numpy as np

# PASO 2.1: CARGAR Y RECORTAR LA IMAGEN ORIGINAL
# Crear carpetas de resultados
Path("resultados").mkdir(parents=True, exist_ok=True)

# Cargar imagen
img = cv2.imread("imagenes/imagen1.png")

if img is None:
    raise FileNotFoundError("No se ha encontrado la imagen.")

# Obtener dimensiones
alto, ancho = img.shape[:2]

print("Alto:", alto)
print("Ancho:", ancho)

# Recortar la franja superior
recorte = img[25:alto, 0:ancho]

# Guardar imagen recortada
cv2.imwrite("resultados/imagen_recortada.png", recorte)

# PASO 2.2: PREPROCESAMIENTO CON OPENCV
# Convertir imagen recortada a escala de grises
gris = cv2.cvtColor(recorte, cv2.COLOR_BGR2GRAY)

# Aumentar resolución al doble
ampliada = cv2.resize(
    gris,
    None,
    fx=2,
    fy=2,
    interpolation=cv2.INTER_CUBIC
)

# Mejorar contraste local mediante CLAHE
clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)

contraste = clahe.apply(ampliada)

# Aplicar filtro de nitidez (Sharpening)
kernel = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
], dtype="float32")

nitidez = cv2.filter2D(
    contraste,
    -1,
    kernel
)

# Aplicar umbralización de Otsu
_, otsu = cv2.threshold(
    contraste,
    0,
    255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

# Aplicar umbralización adaptativa
adaptativa = cv2.adaptiveThreshold(
    contraste,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    31,
    10
)

# PASO 2.3: GUARDAR LAS IMÁGENES PROCESADAS

imagenes_procesadas = {
    "imagen_ampliada": ampliada,
    "imagen_clahe": contraste,
    "imagen_nitidez": nitidez,
    "imagen_otsu": otsu,
    "imagen_adaptativa": adaptativa
}

for nombre, imagen in imagenes_procesadas.items():

    plt.figure(figsize=(14, 8))
    plt.imshow(imagen, cmap="gray")
    plt.title(nombre)
    plt.axis("off")

    plt.savefig(
        f"resultados/{nombre}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

print("Todas las imágenes se han guardado correctamente.")

# PASO 3.1: DETECCIÓN DE LA ESTRUCTURA DE LA TABLA

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

    # Comprobar que se han detectado las divisiones
    if len(filas) < 3 or len(columnas) < 4:
        raise ValueError(
            "No se han detectado correctamente las líneas de la tabla."
        )

    print("\nLíneas horizontales detectadas:", len(filas))
    print("Líneas verticales detectadas:", len(columnas))

    return filas, columnas


# Detectar la tabla utilizando la imagen ampliada
filas, columnas = detectar_lineas_tabla(ampliada)

# PASO 3.2: VISUALIZAR LAS LÍNEAS DETECTADAS

tabla_visual = cv2.cvtColor(
    ampliada,
    cv2.COLOR_GRAY2BGR
)

# Dibujar líneas horizontales
for y in filas:

    cv2.line(
        tabla_visual,
        (0, y),
        (tabla_visual.shape[1], y),
        (0, 0, 255),
        2
    )

# Dibujar líneas verticales
for x in columnas:

    cv2.line(
        tabla_visual,
        (x, 0),
        (x, tabla_visual.shape[0]),
        (0, 255, 0),
        2
    )

plt.figure(figsize=(14, 8))

plt.imshow(
    cv2.cvtColor(tabla_visual, cv2.COLOR_BGR2RGB)
)

plt.title("Detección de líneas de la tabla")
plt.axis("off")

plt.savefig(
    "resultados/deteccion_tabla.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# PASO 3.3: CONFIGURACIÓN DE PYTESSERACT

# pytesseract.pytesseract.tesseract_cmd = (
#     r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# )

carpeta_textos = Path("resultados/ocr")
carpeta_textos.mkdir(parents=True, exist_ok=True)

# PASO 3.4: RECORTAR LAS CELDAS Y EXTRAER EL TEXTO

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

    return texto

# PASO 3.5: OCR DE LAS CINCO IMÁGENES

for nombre, imagen in imagenes_procesadas.items():

    print(f"\nProcesando: {nombre}")

    datos_tabla = []

    # Recorrer todas las filas detectadas,
    # incluyendo la primera fila (encabezado)

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

        # Guardar los tres campos de la fila
        datos_tabla.append([
            metodo,
            endpoint,
            proposito
        ])

    
    # GUARDAR RESULTADO OCR EN TXT
    archivo_txt = carpeta_textos / f"{nombre}.txt"

    with open(
        archivo_txt,
        "w",
        encoding="utf-8"
    ) as archivo:

        for fila in datos_tabla:

            archivo.write(
                f"{fila[0]}\t{fila[1]}\t{fila[2]}\n"
            )

    print(f"TXT generado: {archivo_txt}")


print("\nProceso OCR finalizado.")



