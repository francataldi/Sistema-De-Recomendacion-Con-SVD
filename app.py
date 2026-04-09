import os
import pickle

import numpy as np
import pandas as pd
import streamlit as st

# ── Configuración de la página ───────────────────────────────
st.set_page_config(
    page_title="Recomendador de Películas",
    page_icon="🎬",
    layout="centered"
)

@st.cache_resource
def cargar_modelo():
    # Obtiene la ruta de la carpeta donde está ESTE archivo app.py
    ruta_actual = os.path.dirname(__file__)
    # Construye la ruta absoluta al modelo de forma inteligente
    ruta_pkl = os.path.join(ruta_actual, 'modelo', 'modelo_hibrido.pkl')
    
    with open(ruta_pkl, 'rb') as f:
        return pickle.load(f)

# ── Cargar modelo y desempaquetar variables globales ─────────
modelo       = cargar_modelo()
matriz_aprox = modelo['matriz_aprox']
similitud_df = modelo['similitud_df']
movies       = modelo['movies']
matriz       = modelo['matriz']
ALPHA        = modelo['alpha']

# ── Funciones del sistema híbrido ────────────────────────────

def calcular_score_contenido(movieId, peliculas_vistas):
    """
    Calcula qué tan similar es una película a todo lo que el usuario vio,
    ponderando por el rating que les dio.
    """
    sims = similitud_df.loc[movieId, peliculas_vistas.index]
    score = np.dot(sims.values, peliculas_vistas.values) / (peliculas_vistas.values.sum() + 1e-8)
    return score


def recomendar(userId, alpha=ALPHA, n=10):
    """
    Dado un userId, devuelve las n películas más recomendadas
    combinando filtrado colaborativo (SVD) y filtrado por contenido.
    """
    if userId not in matriz.index:
        return None

    peliculas_vistas    = matriz.loc[userId].dropna()
    peliculas_no_vistas = [m for m in matriz.columns if m not in peliculas_vistas.index]

    # Score colaborativo
    score_colab = matriz_aprox.loc[userId, peliculas_no_vistas]

    # Score de contenido (vectorizado)
    cols_vistas    = [c for c in similitud_df.columns if c in peliculas_vistas.index]
    ratings_vistos = peliculas_vistas[cols_vistas].values
    sims_matrix    = similitud_df.loc[peliculas_no_vistas, cols_vistas].values
    scores_cont    = sims_matrix @ ratings_vistos / (ratings_vistos.sum() + 1e-8)
    score_cont     = pd.Series(scores_cont, index=peliculas_no_vistas)

    # Normalizar ambos a [0, 1]
    def normalizar(s):
        return (s - s.min()) / (s.max() - s.min() + 1e-8)

    score_final = alpha * normalizar(score_colab) + (1 - alpha) * normalizar(score_cont)

    top_n = score_final.sort_values(ascending=False).head(n)

    resultado = pd.DataFrame({
        'movieId':      top_n.index,
        'score_final':  top_n.values
    }).merge(movies[['movieId', 'title']], on='movieId')

    return resultado[['title', 'score_final']]


# ── Interfaz ─────────────────────────────────────────────────

st.title("🎬 Recomendador de Películas")
st.markdown(
    "Sistema de recomendación híbrido que combina "
    "**Filtrado Colaborativo (SVD)** y **Filtrado por Contenido**. "
    "Ingresá tu ID de usuario para ver tus recomendaciones personalizadas."
)

st.divider()

# Selector de usuario
userId = st.number_input(
    label="ID de usuario (entre 1 y 943)",
    min_value=1,
    max_value=943,
    value=1,
    step=1
)

# Slider de alpha (opcional — para que el usuario experimente)
with st.expander("⚙️ Configuración avanzada"):
    alpha = st.slider(
        label="Alpha — peso del modelo colaborativo",
        min_value=0.0,
        max_value=1.0,
        value=float(ALPHA),
        step=0.1,
        help="0 = solo contenido | 1 = solo colaborativo | 0.7 = valor óptimo"
    )

# Mostrar películas que ya vio el usuario
if userId in matriz.index:
    peliculas_vistas = matriz.loc[userId].dropna()
    titulos_vistos   = movies[movies['movieId'].isin(peliculas_vistas.index)][['movieId', 'title']].copy()
    titulos_vistos['rating'] = titulos_vistos['movieId'].map(peliculas_vistas)
    titulos_vistos   = titulos_vistos.sort_values('rating', ascending=False)

    with st.expander(f"🎞️ Películas que vio el usuario {userId} ({len(peliculas_vistas)} en total)"):
        st.dataframe(
            titulos_vistos[['title', 'rating']].rename(columns={'title': 'Título', 'rating': 'Rating'}),
            hide_index=True,
            use_container_width=True
        )

# Botón de recomendación
if st.button("🔍 Ver recomendaciones", type="primary"):
    with st.spinner("Calculando recomendaciones..."):
        recomendaciones = recomendar(userId=userId, alpha=alpha, n=10)

    if recomendaciones is None:
        st.error(f"El usuario {userId} no existe en el dataset.")
    else:
        st.subheader(f"Top 10 recomendaciones para el usuario {userId}")

        # Mostrar como tabla limpia con barras de score
        st.dataframe(
            recomendaciones.rename(columns={
                'title':       'Película',
                'score_final': 'Score'
            }).assign(Score=lambda df: df['Score'].round(3)),
            hide_index=True,
            use_container_width=True,
            column_config={
                "Score": st.column_config.ProgressColumn(
                    "Score",
                    min_value=0,
                    max_value=1,
                    format="%.3f"
                )
            }
        )

st.divider()
st.caption(
    "Proyecto de portafolio · Licenciatura en Ciencias de Datos · UBA · "
    "Dataset: MovieLens 100K"
)
