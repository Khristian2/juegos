import streamlit as st
import pandas as pd
import pickle
import os
from PIL import Image

# --- Estilo Personalizado ---
st.markdown(
    """
    <style>
    body {
        font-family: sans-serif;
        background-color: #111827; /* Fondo Oscuro */
        color: white;
        margin: 0;
    }
    .stApp {
        max-width: none !important;
        margin: 0 !important;
        padding: 2rem;
    }
    .st-container {
        background-color: #fff !important;
        color: #000 !important;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .st-header h1, .st-subheader {
        color: #007bff !important;
        text-align: center;
    }
    label {
        color: white !important;
        font-weight: bold;
        display: block;
        margin-bottom: 0.5rem;
    }
    .st-selectbox div > div > div > div,
    .st-slider div > div > div,
    .st-number-input div > div > input {
        border: 1px solid #ddd;
        border-radius: 4px;
        padding: 0.75rem;
        color: #000;
        background-color: #f9f9f9;
        margin-bottom: 1rem;
    }
    .st-button > button {
        background-color: #007bff;
        color: white !important;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: bold;
        border: none;
        cursor: pointer;
        transition: background-color 0.3s ease;
        margin-top: 1rem;
    }
    .st-button > button:hover {
        background-color: #0056b3;
    }
    .prediction-result {
        font-size: 1.2rem;
        font-weight: bold;
        margin-top: 1.5rem;
        text-align: center;
    }
    .average-spend {
        color: #28a745;
        font-weight: bold;
    }
    .model-info {
        margin-top: 1rem;
        font-style: italic;
        color: #999;
    }
    .st-slider > div > div > div > div[data-testid="stTrack"] {
        background-color: #007bff;
    }
    .st-slider > div > div > div > div[data-testid="stThumb"] {
        background-color: #007bff;
    }
    .st-sidebar {
        background-color: #111827 !important;
        color: white !important;
        padding-top: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Cargar el Modelo Pre-entrenado ---
try:
    with open('modelo-reg-tree-knn-nn.pkl', 'rb') as file:
        model_Tree, model_Knn, model_NN, variables, min_max_scaler = pickle.load(file)
except FileNotFoundError:
    st.error("No se encontró el archivo del modelo 'modelo-reg-tree-knn-nn.pkl'. Asegúrate de que esté en la misma carpeta.")
    st.stop()
except Exception as e:
    st.error(f"Ocurrió un error al cargar el modelo: {e}")
    st.stop()

# --- Contenido Principal ---
st.container()

# --- Logo y Título ---
st.image("1.jpg", caption="Predicción Inteligente de Gasto Gamer", width=150, use_container_width=True)
st.header("Predicción Inteligente de Gasto Gamer")
st.subheader("Conoce cuánto podrías invertir en videojuegos")

# --- Campos de Entrada del Usuario ---
with st.sidebar:
    st.header("Tu Perfil Gamer")
    edad = st.slider("¿Cuántos años tienes?:", min_value=14, max_value=120, value=30, step=1)
    genero = st.selectbox("¿Cuál es tu género?:", ["Hombre", "Mujer", "Otro"])
    tipo_juego = st.selectbox(
        "¿Qué género de videojuego te gusta más?:",
        ["RPG de Acción Épica", "Simulación Estratégica Urbana", "Survival Horror Inmersivo", "Shooter en Primera Persona Competitivo", "Deportes y Simulación Realista", "Carreras de Alta Velocidad", "Aventura de Mundo Abierto Fantástica", "Ciencia Ficción y Disparos Tácticos"]
    )
    plataforma = st.selectbox(
        "¿En qué plataforma prefieres jugar?:",
        ["Ordenador (PC)", "Xbox", "Play Station", "Otras"]
    )
    frecuencia_juego = st.selectbox("¿Con qué frecuencia juegas?:", ["Diariamente", "Semanalmente", "Mensualmente", "Rara vez"])

    # --- Botón para Realizar la Predicción ---
    boton_predecir = st.button("Calcular Estimación de Gasto")

if boton_predecir:
    # Crear un DataFrame con los datos de entrada del usuario
    data_usuario = pd.DataFrame({
        'Edad': [edad],
        'videojuego': [tipo_juego],
        'Plataforma': [plataforma],
        'Sexo': [genero],
        'Consumidor_habitual': [frecuencia_juego] # Usamos la nueva variable
    })

    # Mostrar los datos ingresados en una tabla
    st.subheader("Tus Datos:")
    st.table(data_usuario)

    # **Preprocesamiento de datos para el modelo**
    data_preparada = data_usuario.copy()
    data_preparada = pd.get_dummies(data_preparada, columns=['videojuego'], prefix='videojuego')
    data_preparada = pd.get_dummies(data_preparada, columns=['Plataforma'], prefix='Plataforma')
    data_preparada = pd.get_dummies(data_preparada, columns=['Sexo'], prefix='Sexo')
    data_preparada = pd.get_dummies(data_preparada, columns=['Consumidor_habitual'], prefix='Consumidor_habitual')
    data_preparada = data_preparada.reindex(columns=variables, fill_value=0)
    data_preparada[['Edad']] = min_max_scaler.transform(data_preparada[['Edad']])

    try:
        # Realizar la predicción con los tres modelos
        prediccion_tree = model_Tree.predict(data_preparada.values)[0]
        prediccion_knn = model_Knn.predict(data_preparada.values)[0]
        prediccion_nn = model_NN.predict(data_preparada.values)[0]

        # Calcular el promedio de las predicciones
        prediccion_promedio = (prediccion_tree + prediccion_knn + prediccion_nn) / 3

        # Mostrar el resultado de la predicción promediada
        st.subheader("Estimación de Gasto Gamer")
        st.markdown(f"<p class='prediction-result average-spend'>Según tu perfil, podrías gastar alrededor de: <b>${prediccion_promedio:.2f}</b> en videojuegos bajo estas condiciones.</p>", unsafe_allow_html=True)
        st.markdown(f"<p class='model-info'>Modelos utilizados: Árbol de Decisión, KNN, Red Neuronal</p>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Ocurrió un error durante la predicción: {e}")