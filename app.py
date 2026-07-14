import json

import numpy as np
import streamlit as st
from PIL import Image
from scipy import ndimage
from streamlit_drawable_canvas import st_canvas
from tensorflow import keras

st.set_page_config(page_title="Clasificador de Dígitos", layout="centered")


# ---------------------------------------------------------------------------
# Estilos
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(180deg, #0f1220 0%, #1a1f38 100%);
        }
        .main-title {
            font-size: 2.4rem;
            font-weight: 800;
            color: #f5f5ff;
            text-align: center;
            margin-bottom: 0.2rem;
            letter-spacing: -0.5px;
        }
        .subtitle {
            text-align: center;
            color: #9aa0c3;
            font-size: 1rem;
            margin-bottom: 1.8rem;
        }
        .card {
            background: #171c30;
            border: 1px solid #2b3156;
            border-radius: 16px;
            padding: 1.4rem;
            box-shadow: 0 6px 20px rgba(0,0,0,0.25);
        }
        .result-box {
            background: linear-gradient(135deg, #2c1b52, #171c30);
            border: 1px solid #6d4dc9;
            border-radius: 16px;
            padding: 1.6rem;
            text-align: center;
            margin-top: 1rem;
        }
        .result-number {
            font-size: 3.4rem;
            font-weight: 900;
            color: #ffffff;
            letter-spacing: 6px;
        }
        .digit-badge {
            display: inline-block;
            background: #262c4d;
            border: 1px solid #4a4f80;
            border-radius: 10px;
            padding: 6px 14px;
            margin: 4px;
            font-size: 1.1rem;
            color: #d8dbf5;
        }
        .digit-badge span {
            color: #8ef0c0;
            font-weight: 700;
        }
        .stButton>button {
            background: #6d4dc9;
            color: white;
            border-radius: 10px;
            border: none;
            padding: 0.6rem 1.6rem;
            font-weight: 600;
            width: 100%;
        }
        .stButton>button:hover {
            background: #7f5ee0;
            color: white;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def cargar_modelo():
    modelo = keras.models.load_model("mnist_digit_classifier.keras")
    with open("class_names.json") as f:
        clases = json.load(f)
    return modelo, clases


model, class_names = cargar_modelo()


# ---------------------------------------------------------------------------
# Segmentación y preprocesamiento
# ---------------------------------------------------------------------------
def binarizar(imagen_rgba):
    """RGBA del canvas -> array 2D en escala de grises, fondo negro / trazo blanco."""
    img = Image.fromarray(imagen_rgba.astype("uint8"), mode="RGBA").convert("L")
    arr = np.array(img)
    if arr.mean() > 127:
        arr = 255 - arr
    return arr


def centrar_en_28x28(recorte):
    """Redimensiona un recorte manteniendo proporción y lo centra en un lienzo 28x28,
    igual que el preprocesamiento estándar de MNIST."""
    h, w = recorte.shape
    escala = 20.0 / max(h, w)
    nuevo_h, nuevo_w = max(1, int(h * escala)), max(1, int(w * escala))
    digito = np.array(Image.fromarray(recorte).resize((nuevo_w, nuevo_h)))

    lienzo = np.zeros((28, 28), dtype=np.float32)
    y_off = (28 - nuevo_h) // 2
    x_off = (28 - nuevo_w) // 2
    lienzo[y_off:y_off + nuevo_h, x_off:x_off + nuevo_w] = digito
    return lienzo / 255.0


def segmentar_digitos(arr_binario, area_minima=40):
    """Encuentra cada número dibujado como una región separada y los devuelve
    ordenados de izquierda a derecha."""
    mascara = arr_binario > 30
    estructura = np.ones((3, 3))  # conecta también en diagonal
    etiquetas, n = ndimage.label(mascara, structure=estructura)

    recortes = []
    for objeto, caja in zip(range(1, n + 1), ndimage.find_objects(etiquetas)):
        if caja is None:
            continue
        region = arr_binario[caja] * (etiquetas[caja] == objeto)
        if (region > 30).sum() < area_minima:
            continue  # ruido/punto accidental
        x_inicio = caja[1].start
        recortes.append((x_inicio, region))

    recortes.sort(key=lambda t: t[0])
    return [centrar_en_28x28(r) for _, r in recortes]


def predecir(imagen_rgba):
    arr = binarizar(imagen_rgba)
    digitos = segmentar_digitos(arr)

    resultados = []
    for digito in digitos:
        entrada = np.expand_dims(digito, axis=0)
        pred = model.predict(entrada, verbose=0)[0]
        indice = int(np.argmax(pred))
        resultados.append((class_names[indice], float(pred[indice]), pred))
    return resultados


# ---------------------------------------------------------------------------
# Interfaz
# ---------------------------------------------------------------------------
st.markdown('<div class="main-title">Clasificador de Dígitos</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Dibuja uno o varios números separados entre sí y presiona Predecir</div>',
    unsafe_allow_html=True,
)

col_izq, col_der = st.columns([1, 1], gap="large")

with col_izq:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 1)",
        stroke_width=18,
        stroke_color="#000000",
        background_color="#FFFFFF",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )
    predecir_click = st.button("Predecir")
    st.markdown("</div>", unsafe_allow_html=True)

with col_der:
    if predecir_click:
        if canvas_result.image_data is not None and canvas_result.image_data.sum() > 0:
            resultados = predecir(canvas_result.image_data)

            if not resultados:
                st.warning("No se detectó ningún número. Intenta dibujar con trazo más grueso.")
            else:
                numero_completo = "".join(r[0] for r in resultados)
                st.markdown(
                    f'<div class="result-box">'
                    f'<div style="color:#9aa0c3; font-size:0.9rem;">RESULTADO</div>'
                    f'<div class="result-number">{numero_completo}</div>'
                    f"</div>",
                    unsafe_allow_html=True,
                )

                st.write("")
                badges = "".join(
                    f'<span class="digit-badge">Dígito {i+1}: <span>{d}</span> '
                    f'({conf*100:.0f}%)</span>'
                    for i, (d, conf, _) in enumerate(resultados)
                )
                st.markdown(badges, unsafe_allow_html=True)

                if len(resultados) == 1:
                    st.write("")
                    st.caption("Distribución de probabilidad por clase")
                    _, _, distrib = resultados[0]
                    st.bar_chart(
                        {class_names[i]: float(distrib[i]) for i in range(len(class_names))}
                    )
        else:
            st.info("Dibuja al menos un número antes de predecir.")
    else:
        st.markdown(
            '<div class="card" style="text-align:center; color:#6b7099; padding-top:3rem; padding-bottom:3rem;">'
            "Tu resultado va a aparecer aquí"
            "</div>",
            unsafe_allow_html=True,
        )

st.markdown(
    '<p style="text-align:center; color:#4a4f70; font-size:0.8rem; margin-top:2rem;">'
    "Red neuronal entrenada con TensorFlow/Keras sobre el dataset MNIST"
    "</p>",
    unsafe_allow_html=True,
)

