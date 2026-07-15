import json

import numpy as np
import streamlit as st
from PIL import Image
from scipy import ndimage
from streamlit_drawable_canvas import st_canvas
from tensorflow import keras

st.set_page_config(page_title="Clasificador de Dígitos", layout="centered")


# ---------------------------------------------------------------------------
# Estilos — concepto: hoja cuadriculada de cuaderno / ficha de examen
# Paleta: crema #F5F5EB, beige #EFE7DA / #E1DACA, taupe #C1B6A3, arcilla #B3907A
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Sora:wght@400;500;600;700&display=swap');

        .stApp {
            background-color: #F5F5EB;
            background-image:
                linear-gradient(#E1DACA 1px, transparent 1px),
                linear-gradient(90deg, #E1DACA 1px, transparent 1px);
            background-size: 26px 26px;
        }

        html, body, [class*="css"] {
            font-family: 'Sora', sans-serif;
        }

        .eyebrow {
            font-family: 'Space Mono', monospace;
            font-size: 0.72rem;
            letter-spacing: 3px;
            color: #B3907A;
            text-transform: uppercase;
            text-align: center;
            margin-bottom: 0.3rem;
        }
        .main-title {
            font-size: 2.5rem;
            font-weight: 700;
            color: #4A3F35;
            text-align: center;
            margin-bottom: 0.2rem;
            letter-spacing: -0.5px;
        }
        .subtitle {
            text-align: center;
            color: #8a7f70;
            font-size: 0.95rem;
            margin-bottom: 2rem;
        }

        .sheet {
            background: #FFFFFF;
            border: 1.5px solid #E1DACA;
            border-radius: 4px;
            padding: 1.5rem 1.5rem 1.2rem 1.5rem;
            position: relative;
            box-shadow: 4px 4px 0px #EFE7DA;
        }
        .sheet-label {
            font-family: 'Space Mono', monospace;
            font-size: 0.7rem;
            letter-spacing: 2px;
            color: #C1B6A3;
            text-transform: uppercase;
            margin-bottom: 0.8rem;
        }

        .ticket {
            background: #4A3F35;
            border-radius: 4px;
            padding: 1.8rem 1.5rem;
            text-align: center;
            margin-top: 1rem;
            position: relative;
            overflow: hidden;
        }
        .ticket::before {
            content: "";
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 4px;
            background: repeating-linear-gradient(
                90deg, #B3907A 0px, #B3907A 8px, transparent 8px, transparent 16px
            );
        }
        .ticket-label {
            font-family: 'Space Mono', monospace;
            font-size: 0.7rem;
            letter-spacing: 3px;
            color: #C1B6A3;
            text-transform: uppercase;
        }
        .ticket-number {
            font-family: 'Space Mono', monospace;
            font-size: 3.6rem;
            font-weight: 700;
            color: #F5F5EB;
            letter-spacing: 10px;
            margin: 0.3rem 0 0 10px;
        }

        .stamp {
            display: inline-block;
            background: #FFFFFF;
            border: 1.5px dashed #C1B6A3;
            border-radius: 6px;
            padding: 6px 12px;
            margin: 4px;
            font-family: 'Space Mono', monospace;
            font-size: 0.9rem;
            color: #8a7f70;
        }
        .stamp b {
            color: #B3907A;
            font-size: 1.1rem;
        }

        .placeholder {
            text-align: center;
            color: #b8ae9e;
            font-family: 'Space Mono', monospace;
            font-size: 0.85rem;
            padding: 3.2rem 1rem;
            border: 1.5px dashed #E1DACA;
            border-radius: 4px;
        }

        .stButton>button {
            background: #B3907A;
            color: #FFFFFF;
            border-radius: 4px;
            border: none;
            padding: 0.6rem 1.6rem;
            font-weight: 600;
            width: 100%;
            font-family: 'Sora', sans-serif;
            letter-spacing: 0.5px;
        }
        .stButton>button:hover {
            background: #9c7a63;
            color: #FFFFFF;
        }

        footer, .footnote {
            text-align: center;
            color: #b8ae9e;
            font-size: 0.78rem;
            margin-top: 2rem;
            font-family: 'Space Mono', monospace;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# El componente de dibujo estira su iframe al ancho de la columna aunque el
# lienzo real sea más angosto; este script lo corrige activamente (el CSS solo
# no basta porque el componente reajusta su propio tamaño tras cargar) y de
# paso dibuja 2 líneas punteadas (overlay, no forma parte del lienzo real)
# que dividen el área en 3 carriles para escribir hasta 3 dígitos.
CANVAS_ANCHO = 300
CANVAS_ALTO = 280

st.components.v1.html(
    f"""
    <script>
    function fixCanvasWidth() {{
        const doc = window.parent.document;
        const iframe = doc.querySelector('iframe[data-testid="stCustomComponentV1"]');
        if (iframe) {{
            iframe.style.width = '{CANVAS_ANCHO}px';
            iframe.style.maxWidth = '{CANVAS_ANCHO}px';
            iframe.style.display = 'block';
            const wrapper = iframe.closest('div[data-testid="stElementContainer"]') || iframe.parentElement;
            if (wrapper) {{
                wrapper.style.width = 'fit-content';
                wrapper.style.position = 'relative';

                if (!wrapper.querySelector('.guia-digitos')) {{
                    const overlay = doc.createElement('div');
                    overlay.className = 'guia-digitos';
                    overlay.style.position = 'absolute';
                    overlay.style.top = '0';
                    overlay.style.left = '0';
                    overlay.style.width = '{CANVAS_ANCHO}px';
                    overlay.style.height = '{CANVAS_ALTO}px';
                    overlay.style.pointerEvents = 'none';
                    overlay.style.zIndex = '10';

                    const tercio = {CANVAS_ANCHO} / 3;
                    [tercio, tercio * 2].forEach((x) => {{
                        const linea = doc.createElement('div');
                        linea.style.position = 'absolute';
                        linea.style.top = '0';
                        linea.style.left = x + 'px';
                        linea.style.width = '0';
                        linea.style.height = '100%';
                        linea.style.borderLeft = '1.5px dashed #C1B6A3';
                        linea.style.opacity = '0.55';
                        overlay.appendChild(linea);
                    }});

                    wrapper.appendChild(overlay);
                }}
            }}
        }}
    }}
    fixCanvasWidth();
    setInterval(fixCanvasWidth, 400);
    </script>
    """,
    height=0,
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
st.markdown('<div class="eyebrow">Reconocimiento de escritura</div>', unsafe_allow_html=True)
st.markdown('<div class="main-title">Clasificador de Dígitos</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Dibuja hasta 3 números, uno por cada carril punteado, y presiona Predecir</div>',
    unsafe_allow_html=True,
)

col_izq, col_der = st.columns([1, 1], gap="large")

with col_izq:
    st.markdown('<div class="sheet-label">Entrada — dibuja aquí</div>', unsafe_allow_html=True)
    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 1)",
        stroke_width=18,
        stroke_color="#4A3F35",
        background_color="#FFFFFF",
        height=CANVAS_ALTO,
        width=CANVAS_ANCHO,
        drawing_mode="freedraw",
        display_toolbar=False,
        key="canvas",
    )
    predecir_click = st.button("Predecir")

with col_der:
    if predecir_click:
        if canvas_result.image_data is not None and canvas_result.image_data.sum() > 0:
            resultados = predecir(canvas_result.image_data)

            if not resultados:
                st.warning("No se detectó ningún número. Intenta dibujar con trazo más grueso.")
            else:
                numero_completo = "".join(r[0] for r in resultados)
                st.markdown(
                    f'<div class="ticket">'
                    f'<div class="ticket-label">Salida</div>'
                    f'<div class="ticket-number">{numero_completo}</div>'
                    f"</div>",
                    unsafe_allow_html=True,
                )

                st.write("")
                stamps = "".join(
                    f'<span class="stamp">Dígito {i+1} · <b>{d}</b> · {conf*100:.0f}%</span>'
                    for i, (d, conf, _) in enumerate(resultados)
                )
                st.markdown(stamps, unsafe_allow_html=True)

                if len(resultados) == 1:
                    st.write("")
                    st.markdown(
                        '<div class="sheet-label" style="margin-top:1rem;">Distribución por clase</div>',
                        unsafe_allow_html=True,
                    )
                    _, _, distrib = resultados[0]
                    st.bar_chart(
                        {class_names[i]: float(distrib[i]) for i in range(len(class_names))}
                    )
        else:
            st.info("Dibuja al menos un número antes de predecir.")
    else:
        st.markdown(
            '<div class="placeholder">esperando trazo &gt;&gt;&gt;</div>',
            unsafe_allow_html=True,
        )

st.markdown(
    '<p class="footnote">Gael Alexander Basana Hernandez · dataset MNIST</p>',
    unsafe_allow_html=True,
)
