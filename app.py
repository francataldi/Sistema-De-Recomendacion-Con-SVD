import os
import pickle
import re

import pandas as pd
import requests
import streamlit as st

from recomendador import (
    peliculas_populares,
    recomendar_hibrido,
    recomendar_para_usuario_nuevo,
)

# ── Configuración de la página ───────────────────────────────
st.set_page_config(
    page_title="Recomendador de Películas",
    page_icon="🎬",
    layout="centered"
)

RUTA_MODELO = os.path.join(os.path.dirname(__file__), 'modelo', 'modelo_hibrido.pkl')

# ── Carga del modelo pre-entrenado ───────────────────────────
# La app NO entrena nada: carga el modelo que generó el notebook.
# @st.cache_resource hace que el pickle se lea una sola vez.
if not os.path.exists(RUTA_MODELO):
    st.error(
        "No se encontró el modelo entrenado (`modelo/modelo_hibrido.pkl`).\n\n"
        "Para generarlo, ejecutá el notebook `main.ipynb` de punta a punta "
        "(la última sección guarda el modelo) y volvé a lanzar la app."
    )
    st.stop()


@st.cache_resource
def cargar_modelo():
    with open(RUTA_MODELO, 'rb') as f:
        return pickle.load(f)


datos = cargar_modelo()
modelo_svd = datos['modelo_svd']      # SVD de scikit-surprise ya entrenado
similitud_df = datos['similitud_df']  # similitud coseno entre películas
movies = datos['movies']              # títulos y géneros
ratings = datos['ratings']            # historial de ratings
ALPHA = datos['alpha']                # alpha óptimo (según NDCG@10)
K_FACTORES = datos['k']               # k óptimo (según RMSE de test)


# ── Pósters vía API de TMDb ──────────────────────────────────
# La API key NUNCA va en el código: se lee de st.secrets, que Streamlit
# toma de .streamlit/secrets.toml en local (ignorado por git) o del panel
# de Secrets en Streamlit Cloud. Sin key, la app funciona en modo texto.

def obtener_api_key_tmdb():
    try:
        return st.secrets["TMDB_API_KEY"]
    except Exception:
        # sin secrets.toml o sin la key: modo sin pósters, no es un error
        return None


def limpiar_titulo(titulo):
    """
    Convierte un título de MovieLens en (título limpio, año) para buscarlo
    en TMDb. Ej: "Shawshank Redemption, The (1994)" -> ("The Shawshank
    Redemption", 1994). MovieLens pone el artículo al final y el año entre
    paréntesis; TMDb espera el título natural y el año aparte.
    """
    m = re.search(r'\((\d{4})\)\s*$', titulo)
    anio = int(m.group(1)) if m else None

    limpio = re.sub(r'\s*\(\d{4}\)\s*$', '', titulo)
    # algunos títulos traen un título alternativo entre paréntesis
    # (ej. "Like Water For Chocolate (Como agua para chocolate)")
    limpio = re.sub(r'\s*\([^)]*\)\s*$', '', limpio)
    # artículo al final -> al principio
    m = re.match(r'^(?P<resto>.+),\s+(?P<articulo>The|A|An|La|Le|Les|Los|El|Il|Das|Der|Die)$',
                 limpio)
    if m:
        limpio = f"{m.group('articulo')} {m.group('resto')}"
    return limpio.strip(), anio


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def buscar_poster(titulo, api_key):
    """
    Busca el póster de una película en TMDb y devuelve la URL completa de
    la imagen, o None si no la encuentra o la request falla (nunca tira
    excepción: los pósters son un plus visual, no funcionalidad core).
    El caché de 24h evita repetir llamadas por la misma película (TMDb
    tiene rate limits).
    """
    if not api_key:
        return None
    try:
        query, anio = limpiar_titulo(titulo)
        params = {'api_key': api_key, 'query': query}
        if anio:
            params['year'] = anio
        r = requests.get(
            'https://api.themoviedb.org/3/search/movie',
            params=params, timeout=5
        )
        r.raise_for_status()
        resultados = r.json().get('results', [])
        if resultados and resultados[0].get('poster_path'):
            return f"https://image.tmdb.org/t/p/w342{resultados[0]['poster_path']}"
    except Exception:
        pass
    return None


API_KEY_TMDB = obtener_api_key_tmdb()
if API_KEY_TMDB is None:
    st.info(
        "Sin API key de TMDb configurada — las recomendaciones se muestran "
        "sin póster. (Se configura en `.streamlit/secrets.toml` como "
        "`TMDB_API_KEY`.)"
    )


# ── Lógica del sistema híbrido (vive en recomendador.py) ─────

def recomendar(userId, alpha=ALPHA, n=10):
    """Wrapper: arma el historial del usuario y llama al módulo compartido."""
    historial = ratings[ratings['userId'] == userId].set_index('movieId')['rating']
    if historial.empty:
        return None
    return recomendar_hibrido(
        modelo_svd, similitud_df, movies, historial,
        userId=userId, alpha=alpha, n=n
    )


# ── Interfaz ─────────────────────────────────────────────────

st.title("🎬 Recomendador de Películas")
st.markdown(
    "Sistema de recomendación **híbrido** que combina "
    "Filtrado Colaborativo (SVD) y Filtrado por Contenido. "
    "Ingresá tu ID de usuario y el sistema predice qué películas te van a gustar."
)

st.divider()

modo = st.radio(
    "¿Quién sos?",
    ["Usuario del dataset (ID 1 a 943)", "Usuario nuevo (sin historial)"],
    horizontal=True,
)
es_usuario_nuevo = modo.startswith("Usuario nuevo")

col1, col2 = st.columns([2, 1])
with col1:
    if not es_usuario_nuevo:
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
        f"El valor óptimo ({ALPHA}) fue elegido maximizando NDCG@10 "
        "sobre un conjunto de prueba."
    )
    alpha = st.slider(
        label="Alpha — peso del modelo colaborativo",
        min_value=0.0, max_value=1.0, value=float(ALPHA), step=0.1,
        help=f"0 = solo contenido | 1 = solo colaborativo | {ALPHA} = valor óptimo"
    )
    st.caption(
        f"Con alpha={alpha:.1f}: {int(alpha*100)}% colaborativo, "
        f"{int((1-alpha)*100)}% contenido · Modelo SVD con k={K_FACTORES} factores"
    )

st.divider()


PLACEHOLDER_POSTER = """
<div style="aspect-ratio:2/3; display:flex; align-items:center;
            justify-content:center; background:rgba(128,128,128,0.15);
            border-radius:8px; font-size:2.5rem;">🎬</div>
"""


def mostrar_recomendaciones(recomendaciones, subtitulo):
    st.subheader(f"Top {len(recomendaciones)} recomendaciones")
    st.markdown(subtitulo)

    # buscamos todos los pósters primero (con caché, así solo la primera
    # vez por película le pega de verdad a la API)
    posters = {
        fila['title']: buscar_poster(fila['title'], API_KEY_TMDB)
        for _, fila in recomendaciones.iterrows()
    }

    POR_FILA = 5
    filas = list(recomendaciones.iterrows())
    for inicio in range(0, len(filas), POR_FILA):
        columnas = st.columns(POR_FILA)
        for col, (_, fila) in zip(columnas, filas[inicio:inicio + POR_FILA]):
            with col:
                url = posters.get(fila['title'])
                if url:
                    st.image(url, use_container_width=True)
                else:
                    st.markdown(PLACEHOLDER_POSTER, unsafe_allow_html=True)
                st.markdown(f"**{fila['title']}**")
                st.progress(min(max(float(fila['score_final']), 0.0), 1.0),
                            text=f"Score {fila['score_final']:.3f}")


if es_usuario_nuevo:
    # ── Cold start: usuario sin historial ────────────────────
    # El modelo colaborativo no sabe nada de un usuario que no estaba en el
    # entrenamiento, así que le pedimos que puntúe algunas películas
    # conocidas ("semillas") y recomendamos SOLO con el modelo de contenido.
    st.markdown(
        "Como el sistema todavía no te conoce, punteá **al menos 3** de estas "
        "películas populares (dejá en *Sin puntuar* las que no viste). "
        "Con eso armamos tu perfil de gustos por género."
    )

    populares = peliculas_populares(ratings, movies, n=10)
    opciones = ['Sin puntuar', '1', '2', '3', '4', '5']

    semillas = {}
    for _, fila in populares.iterrows():
        puntaje = st.select_slider(
            fila['title'], options=opciones, value='Sin puntuar',
            key=f"semilla_{fila['movieId']}"
        )
        if puntaje != 'Sin puntuar':
            semillas[fila['movieId']] = int(puntaje)

    st.caption(f"Películas puntuadas: {len(semillas)} (mínimo 3)")

    if st.button("🔍 Ver recomendaciones", type="primary", use_container_width=True):
        if len(semillas) < 3:
            st.warning("Puntuá al menos 3 películas para que podamos recomendarte algo.")
        else:
            with st.spinner("Calculando recomendaciones..."):
                recomendaciones = recomendar_para_usuario_nuevo(
                    similitud_df, movies, semillas, n=n_recomendaciones
                )
            mostrar_recomendaciones(
                recomendaciones,
                "Calculadas **solo con el modelo de contenido** (similitud de "
                "géneros con tus películas semilla). El modelo colaborativo "
                "necesita historial dentro del dataset, así que acá no participa."
            )

else:
    # ── Usuario existente: flujo híbrido normal ──────────────
    vistas_usuario = ratings[ratings['userId'] == userId]
    if not vistas_usuario.empty:
        with st.expander(
            f"🎞️ Historial del usuario {userId} — {len(vistas_usuario)} películas vistas"
        ):
            titulos_vistos = (
                vistas_usuario
                .merge(movies[['movieId', 'title']], on='movieId')
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
            mostrar_recomendaciones(
                recomendaciones,
                f"Calculadas con alpha=**{alpha:.1f}** "
                f"({int(alpha*100)}% colaborativo · {int((1-alpha)*100)}% contenido)"
            )

st.divider()
st.caption(
    "Proyecto de portafolio · Licenciatura en Ciencias de Datos · UBA · "
    "Dataset: [MovieLens 100K](https://grouplens.org/datasets/movielens/100k/)"
)
