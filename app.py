import json

import numpy as np
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from tensorflow import keras

st.set_page_config(page_title="Clasificador de dígitos MNIST", page_icon="🔢")


@st.cache_resource
def cargar_modelo():
    modelo = keras.models.load_model("mnist_digit_classifier.keras")
    with open("class_names.json") as f:
        clases = json.load(f)
    return modelo, clases


model, class_names = cargar_modelo()


def preprocesar(imagen_rgba):
    """
    Recibe el array RGBA (numpy) que devuelve el canvas de Streamlit
    y lo convierte en un array (28, 28) normalizado (0 a 1),
    listo para el modelo.
    """
    img = Image.fromarray(imagen_rgba.astype("uint8"), mode="RGBA")
    img = img.convert("L")  # escala de grises

    arr = np.array(img)

    # El canvas tiene fondo blanco y trazo negro por defecto;
    # MNIST espera fondo negro (0) y trazo blanco (255), así que invertimos.
    if arr.mean() > 127:
        arr = 255 - arr

    img = Image.fromarray(arr).resize((28, 28))
    arr = np.array(img).astype("float32") / 255.0
    return arr


st.title("🔢 Clasificador de dígitos escritos a mano")
st.write("Dibuja un dígito del 0 al 9 en el recuadro y presiona **Predecir**.")

col1, col2 = st.columns(2)

with col1:
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

with col2:
    if st.button("Predecir", type="primary"):
        if canvas_result.image_data is not None and canvas_result.image_data.sum() > 0:
            arr = preprocesar(canvas_result.image_data)
            entrada = np.expand_dims(arr, axis=0)
            pred = model.predict(entrada, verbose=0)[0]

            indice = int(np.argmax(pred))
            st.subheader(f"Predicción: **{class_names[indice]}**")
            st.write(f"Confianza: {pred[indice] * 100:.1f}%")

            st.bar_chart(
                {class_names[i]: float(pred[i]) for i in range(len(class_names))}
            )
        else:
            st.warning("Dibuja un dígito antes de predecir.")

st.divider()
st.caption("Modelo: red densa entrenada con TensorFlow/Keras sobre el dataset MNIST.")
