import os

import numpy as np
import pandas as pd
import streamlit as st
from scipy.sparse.linalg import svds
from sklearn.metrics.pairwise import cosine_similarity

# ── Configuración de la página ───────────────────────────────
st.set_page_config(
    page_title="Recomendador de Películas",
    page_icon="🎬",
    layout="centered"
)

# ── Entrenamiento del modelo ─────────────────────────────────
# @st.cache_resource hace que esto se ejecute UNA sola vez.
# La primera vez que alguien abre la app tarda ~15 segundos.
# Después queda en memoria — las siguientes visitas son instantáneas.
@st.cache_resource
def entrenar_modelo():
    ruta = os.path.dirname(__file__)

    # --- Cargar datos ---
    cols_item = [
        'movieId', 'title', 'release_date', 'video_release', 'imdb_url',
        'unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy',
        'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
        'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
    ]
    ratings = pd.read_csv(
        os.path.join(ruta, 'Data', 'ml-100k', 'u.data'),
        sep='\t', names=['userId', 'movieId', 'rating', 'timestamp']
    )
    movies = pd.read_csv(
        os.path.join(ruta, 'Data', 'ml-100k', 'u.item'),
        sep='|', names=cols_item, encoding='latin-1', usecols=range(24)
    )

    # --- Construir matriz usuario × película ---
    matriz = ratings.pivot_table(
        index='userId', columns='movieId', values='rating'
    )

    # --- Imputar NaN con la media de cada usuario ---
    matriz_final = matriz.apply(lambda row: row.fillna(row.mean()), axis=1)

    # --- Aplicar SVD con k=50 factores latentes ---
    M = matriz_final.values
    U, sigma, Vt = svds(M, k=50)
    sigma_diag = np.diag(sigma)
    M_aprox = U @ sigma_diag @ Vt

    matriz_aprox = pd.DataFrame(
        M_aprox,
        index=matriz_final.index,
        columns=matriz_final.columns
    )

    # --- Calcular similitud coseno entre películas ---
    genre_cols = [
        'unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy',
        'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
        'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
    ]
    genre_matrix = movies.set_index('movieId')[genre_cols].values
    similitud = cosine_similarity(genre_matrix)
    similitud_df = pd.DataFrame(
        similitud,
        index=movies['movieId'].values,
        columns=movies['movieId'].values
    )

    return matriz, matriz_aprox, similitud_df, movies


# ── Cargar / entrenar ────────────────────────────────────────
with st.spinner("⏳ Cargando modelo (solo la primera vez, ~15 segundos)..."):
    matriz, matriz_aprox, similitud_df, movies = entrenar_modelo()

ALPHA = 0.7   # mejor alpha encontrado durante el hyperparameter tuning


# ── Lógica del sistema híbrido ───────────────────────────────

def recomendar(userId, alpha=ALPHA, n=10):
    """
    Devuelve las n películas más recomendadas para un usuario,
    combinando SVD (colaborativo) y similitud coseno (contenido).
    """
    if userId not in matriz.index:
        return None

    peliculas_vistas    = matriz.loc[userId].dropna()
    peliculas_no_vistas = [m for m in matriz.columns if m not in peliculas_vistas.index]

    # Score colaborativo (directo de la matriz aproximada por SVD)
    score_colab = matriz_aprox.loc[userId, peliculas_no_vistas]

    # Score de contenido (vectorizado con multiplicación matricial)
    cols_vistas    = [c for c in similitud_df.columns if c in peliculas_vistas.index]
    ratings_vistos = peliculas_vistas[cols_vistas].values
    sims_matrix    = similitud_df.loc[peliculas_no_vistas, cols_vistas].values
    scores_cont    = sims_matrix @ ratings_vistos / (ratings_vistos.sum() + 1e-8)
    score_cont     = pd.Series(scores_cont, index=peliculas_no_vistas)

    # Normalizar ambos a [0, 1] para combinarlos en la misma escala
    def normalizar(s):
        return (s - s.min()) / (s.max() - s.min() + 1e-8)

    score_final = alpha * normalizar(score_colab) + (1 - alpha) * normalizar(score_cont)

    top_n = score_final.sort_values(ascending=False).head(n)

    return (
        pd.DataFrame({'movieId': top_n.index, 'score_final': top_n.values})
        .merge(movies[['movieId', 'title']], on='movieId')
        [['title', 'score_final']]
    )


# ── Interfaz ─────────────────────────────────────────────────

st.title("🎬 Recomendador de Películas")
st.markdown(
    "Sistema de recomendación **híbrido** que combina "
    "Filtrado Colaborativo (SVD) y Filtrado por Contenido. "
    "Ingresá tu ID de usuario y el sistema predice qué películas te van a gustar."
)

st.divider()

col1, col2 = st.columns([2, 1])
with col1:
    userId = st.number_input(
        label="ID de usuario (1 a 943)",
        min_value=1, max_value=943, value=1, step=1
    )
with col2:
    n_recomendaciones = st.number_input(
        label="Cantidad de recomendaciones",
        min_value=5, max_value=20, value=10, step=5
    )

with st.expander("⚙️ Configuración avanzada"):
    st.markdown(
        "El parámetro **alpha** controla el balance entre los dos modelos. "
        "El valor óptimo (0.7) fue encontrado evaluando el RMSE sobre un conjunto de prueba."
    )
    alpha = st.slider(
        label="Alpha — peso del modelo colaborativo",
        min_value=0.0, max_value=1.0, value=float(ALPHA), step=0.1,
        help="0 = solo contenido | 1 = solo colaborativo | 0.7 = valor óptimo"
    )
    st.caption(
        f"Con alpha={alpha:.1f}: {int(alpha*100)}% colaborativo, "
        f"{int((1-alpha)*100)}% contenido"
    )

st.divider()

if userId in matriz.index:
    peliculas_vistas = matriz.loc[userId].dropna()
    with st.expander(
        f"🎞️ Historial del usuario {userId} — {len(peliculas_vistas)} películas vistas"
    ):
        titulos_vistos = (
            movies[movies['movieId'].isin(peliculas_vistas.index)][['movieId', 'title']]
            .copy()
            .assign(rating=lambda df: df['movieId'].map(peliculas_vistas))
            .sort_values('rating', ascending=False)
        )
        st.dataframe(
            titulos_vistos[['title', 'rating']].rename(
                columns={'title': 'Película', 'rating': 'Rating dado'}
            ),
            hide_index=True, use_container_width=True
        )

if st.button("🔍 Ver recomendaciones", type="primary", use_container_width=True):
    with st.spinner("Calculando recomendaciones..."):
        recomendaciones = recomendar(userId=userId, alpha=alpha, n=n_recomendaciones)

    if recomendaciones is None:
        st.error(f"El usuario {userId} no existe en el dataset.")
    else:
        st.subheader(f"Top {n_recomendaciones} recomendaciones para el usuario {userId}")
        st.markdown(
            f"Calculadas con alpha=**{alpha:.1f}** "
            f"({int(alpha*100)}% colaborativo · {int((1-alpha)*100)}% contenido)"
        )
        st.dataframe(
            recomendaciones
            .rename(columns={'title': 'Película', 'score_final': 'Score'})
            .assign(Score=lambda df: df['Score'].round(3)),
            hide_index=True,
            use_container_width=True,
            column_config={
                "Score": st.column_config.ProgressColumn(
                    "Score", min_value=0, max_value=1, format="%.3f"
                )
            }
        )

st.divider()
st.caption(
    "Proyecto de portafolio · Licenciatura en Ciencias de Datos · UBA · "
    "Dataset: [MovieLens 100K](https://grouplens.org/datasets/movielens/100k/)"
)