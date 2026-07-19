import os
import pickle
import re

import pandas as pd
import requests
import streamlit as st

from recomendador import (
    buscar_peliculas,
    explicar_recomendacion,
    peliculas_populares,
    recomendar_hibrido,
    recomendar_para_usuario_nuevo,
    titulo_normalizado,
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


# la primera carga del pickle tarda 1-2 segundos; después queda cacheado
with st.spinner("Cargando el modelo entrenado..."):
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
    Redemption", 1994). El título limpio lo arma titulo_normalizado()
    (utilidad compartida en recomendador.py); acá solo extraemos además el
    año, que TMDb usa como parámetro aparte.
    """
    m = re.search(r'\((\d{4})\)\s*$', titulo)
    anio = int(m.group(1)) if m else None
    return titulo_normalizado(titulo), anio


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

# ── Configuración (sidebar, para dejar limpio el flujo principal) ──
with st.sidebar:
    st.header("⚙️ Configuración")

    n_recomendaciones = st.number_input(
        label="Cantidad de recomendaciones",
        min_value=5, max_value=20, value=10, step=5
    )

    st.divider()

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


# ── Tags de género ───────────────────────────────────────────
# Un color fijo por género (los badges de Streamlit soportan esta paleta
# limitada, así que algunos géneros comparten color, pero cada género
# tiene SIEMPRE el mismo — la consistencia es lo que permite reconocer
# patrones visuales).
COLOR_GENERO = {
    'Action': 'red',      'Adventure': 'orange', 'Animation': 'violet',
    'Children': 'green',  'Comedy': 'orange',    'Crime': 'gray',
    'Documentary': 'gray', 'Drama': 'blue',      'Fantasy': 'violet',
    'Film-Noir': 'gray',  'Horror': 'red',       'Musical': 'green',
    'Mystery': 'blue',    'Romance': 'red',      'Sci-Fi': 'blue',
    'Thriller': 'orange', 'War': 'gray',         'Western': 'orange',
    'unknown': 'gray',
}
GENERO_COLS = list(COLOR_GENERO.keys())

# géneros y título de cada película, indexados por movieId
generos_por_id = movies.set_index('movieId')[GENERO_COLS]
titulo_de = movies.set_index('movieId')['title']


def badges_de_generos(movieId):
    """Markdown con un badge de color por género de la película."""
    if movieId not in generos_por_id.index:
        return ""
    fila = generos_por_id.loc[movieId]
    generos = [g for g in GENERO_COLS if fila[g] == 1]
    return " ".join(f":{COLOR_GENERO[g]}-badge[{g}]" for g in generos)


def perfil_de_generos(ratings_usuario):
    """
    Rating promedio que el usuario le dio a cada género, según su
    historial (o sus semillas si es un usuario nuevo). Devuelve una Series
    género -> promedio, solo con los géneros de los que hay algún dato,
    o None si no hay nada que graficar.
    """
    ids = generos_por_id.index.intersection(ratings_usuario.index)
    if len(ids) == 0:
        return None
    generos_vistos = generos_por_id.loc[ids]

    promedios = {}
    for g in GENERO_COLS:
        if g == 'unknown':
            continue
        ids_genero = generos_vistos.index[generos_vistos[g] == 1]
        if len(ids_genero) > 0:
            promedios[g] = ratings_usuario.loc[ids_genero].mean()
    if not promedios:
        return None
    return pd.Series(promedios).sort_values(ascending=False)


def mostrar_perfil_de_generos(ratings_usuario, titulo_expander):
    perfil = perfil_de_generos(ratings_usuario)
    if perfil is None:
        return
    with st.expander(titulo_expander):
        st.caption(
            "Rating promedio que le diste a cada género. Es el 'perfil de "
            "gustos' que el modelo de contenido usa para recomendarte."
        )
        st.bar_chart(
            perfil, horizontal=True,
            height=max(220, 26 * len(perfil)),
            x_label="Rating promedio", y_label="",
        )


def explicaciones_para(recomendaciones, ratings_usuario, por_peso=False):
    """
    Arma el texto de 'por qué te recomiendo esto' para cada película del
    Top-N, usando explicar_recomendacion() del módulo compartido.
    Devuelve un dict movieId -> texto.
    """
    textos = {}
    for m in recomendaciones['movieId']:
        exp = explicar_recomendacion(similitud_df, ratings_usuario, m,
                                     por_peso=por_peso)
        if exp is None:
            continue
        mid_similar, _, rating = exp
        if por_peso:
            textos[m] = (f"Tu semilla que más pesó: **{titulo_de[mid_similar]}** "
                         f"(le pusiste {int(rating)}★)")
        else:
            textos[m] = (f"Similar a **{titulo_de[mid_similar]}** "
                         f"(que calificaste con {int(rating)}★)")
    return textos


PLACEHOLDER_POSTER = """
<div style="aspect-ratio:2/3; display:flex; align-items:center;
            justify-content:center; background:rgba(128,128,128,0.15);
            border-radius:8px; font-size:2.5rem;">🎬</div>
"""


def mostrar_recomendaciones(recomendaciones, subtitulo, explicaciones=None):
    st.subheader(f"Top {len(recomendaciones)} recomendaciones")
    st.markdown(subtitulo)
    explicaciones = explicaciones or {}

    # buscamos todos los pósters primero (con caché, así solo la primera
    # vez por película le pega de verdad a la API). Las llamadas a TMDb
    # son la única espera real acá, por eso el spinner va solo si hay key.
    if API_KEY_TMDB:
        with st.spinner("Buscando pósters en TMDb..."):
            posters = {
                fila['title']: buscar_poster(fila['title'], API_KEY_TMDB)
                for _, fila in recomendaciones.iterrows()
            }
    else:
        posters = {}

    POR_FILA = 5
    filas = list(recomendaciones.iterrows())
    for inicio in range(0, len(filas), POR_FILA):
        columnas = st.columns(POR_FILA)
        for col, (_, fila) in zip(columnas, filas[inicio:inicio + POR_FILA]):
            with col:
                url = posters.get(fila['title'])
                if url:
                    st.image(url, width='stretch')
                else:
                    st.markdown(PLACEHOLDER_POSTER, unsafe_allow_html=True)
                st.markdown(f"**{fila['title']}**")
                badges = badges_de_generos(fila['movieId'])
                if badges:
                    st.markdown(badges)
                st.progress(min(max(float(fila['score_final']), 0.0), 1.0),
                            text=f"Score {fila['score_final']:.3f}")
                if fila['movieId'] in explicaciones:
                    st.caption(explicaciones[fila['movieId']])


def quitar_del_carrito(movieId):
    """
    Callback del botón 'Sacar': saca la película del carrito y resetea el
    slider correspondiente a 'No puntuar'. Se hace en un callback (corre
    ANTES del rerun) para poder modificar el estado del widget sin que
    Streamlit se queje de tocarlo después de haberlo instanciado.
    """
    st.session_state['carrito_semillas'].pop(movieId, None)
    clave = f"sem_punt_{movieId}"
    if clave in st.session_state:
        st.session_state[clave] = 'No puntuar'


OPCIONES_PUNTAJE = ['No puntuar', '1', '2', '3', '4', '5']


def mostrar_tarjetas_puntuables(peliculas):
    """
    Muestra películas como tarjetas (póster + título + badges de género)
    con un select_slider debajo para puntuarlas. El carrito de semillas en
    st.session_state es la fuente de verdad: el slider se inicializa desde
    ahí y sincroniza de vuelta en cada rerun, así una puntuación no se
    pierde aunque la película deje de mostrarse (p. ej. al cambiar la
    búsqueda). 'peliculas' es un DataFrame con al menos movieId y title.

    Solo se piden pósters de las películas mostradas acá (<=12), reusando
    buscar_poster() con su caché de 24hs — no hace falta rate-limiting
    manual extra.
    """
    carrito = st.session_state['carrito_semillas']

    if API_KEY_TMDB:
        with st.spinner("Buscando pósters en TMDb..."):
            posters = {
                fila['title']: buscar_poster(fila['title'], API_KEY_TMDB)
                for _, fila in peliculas.iterrows()
            }
    else:
        posters = {}

    POR_FILA = 4
    filas = list(peliculas.iterrows())
    for inicio in range(0, len(filas), POR_FILA):
        columnas = st.columns(POR_FILA)
        for col, (_, fila) in zip(columnas, filas[inicio:inicio + POR_FILA]):
            mid = int(fila['movieId'])
            clave = f"sem_punt_{mid}"
            # sembramos el slider desde el carrito la primera vez que
            # aparece (después su propio estado manda)
            if clave not in st.session_state:
                st.session_state[clave] = (
                    str(carrito[mid]) if mid in carrito else 'No puntuar'
                )
            with col:
                url = posters.get(fila['title'])
                if url:
                    st.image(url, width='stretch')
                else:
                    st.markdown(PLACEHOLDER_POSTER, unsafe_allow_html=True)
                st.markdown(f"**{fila['title']}**")
                badges = badges_de_generos(mid)
                if badges:
                    st.markdown(badges)
                valor = st.select_slider(
                    "Tu puntaje", options=OPCIONES_PUNTAJE, key=clave,
                    label_visibility="collapsed",
                )
                # sincronizamos el carrito con lo que muestra el slider
                if valor == 'No puntuar':
                    carrito.pop(mid, None)
                else:
                    carrito[mid] = int(valor)


# Cada modo vive en su propio tab. OJO: a diferencia del st.radio anterior,
# con tabs AMBOS flujos se renderizan siempre (el tab no activo queda
# oculto), por eso los widgets repetidos llevan key único.
tab_existente, tab_nuevo = st.tabs(["👤 Usuario del dataset", "✨ Usuario nuevo"])

with tab_nuevo:
    # ── Cold start: usuario sin historial ────────────────────
    # El modelo colaborativo no sabe nada de un usuario que no estaba en el
    # entrenamiento, así que le pedimos que puntúe algunas películas
    # conocidas ("semillas") y recomendamos SOLO con el modelo de contenido.
    #
    # NOTA sobre el catálogo: MovieLens 100k no tiene nada posterior a abril
    # de 1998, así que el buscador de abajo NO resuelve "no conozco ninguna
    # película moderna" (no las hay en el dataset). Lo que resuelve es el
    # otro problema: antes puntuabas a ciegas una lista de títulos en texto
    # plano; ahora podés ver el póster y los géneros de cada una para saber
    # de qué trata antes de decidir si la puntuás.
    st.markdown(
        "Como el sistema todavía no te conoce, buscá y puntuá **al menos 3** "
        "películas que hayas visto. Con eso armamos tu perfil de gustos por "
        "género. *(El catálogo es de MovieLens 100k: películas hasta 1998.)*"
    )

    # el carrito de semillas es la fuente de verdad; vive en session_state
    # para no perderse con cada tecleo del buscador (cada uno es un rerun)
    if 'carrito_semillas' not in st.session_state:
        st.session_state['carrito_semillas'] = {}
    carrito = st.session_state['carrito_semillas']

    query = st.text_input(
        "🔎 Buscá una película",
        placeholder="Por ejemplo: 'Star Wars', 'Toy Story', 'Godfather'…",
        key="busqueda_semillas",
    )

    if query.strip():
        resultados = buscar_peliculas(movies, query, n=12)
        if resultados.empty:
            st.info(
                f"No encontré películas con «{query}». Probá con otro título "
                "(recordá que el catálogo llega hasta 1998)."
            )
        else:
            st.caption(f"{len(resultados)} resultado(s) — puntuá las que hayas visto:")
            mostrar_tarjetas_puntuables(resultados)
    else:
        st.caption("Estas son las más populares del dataset — o buscá arriba:")
        populares = peliculas_populares(ratings, movies, n=8)
        mostrar_tarjetas_puntuables(populares)

    # ── Carrito: lo que el usuario ya puntuó (venga de donde venga) ──
    st.divider()
    st.markdown("#### 🛒 Tu selección")
    if carrito:
        for mid in list(carrito.keys()):
            c1, c2, c3 = st.columns([6, 2, 2])
            c1.write(titulo_de[mid])
            c2.write("⭐" * carrito[mid])
            c3.button(
                "❌ Sacar", key=f"quitar_{mid}",
                on_click=quitar_del_carrito, args=(mid,),
            )
    else:
        st.caption("Todavía no puntuaste ninguna película.")

    st.caption(f"Películas puntuadas: {len(carrito)} (mínimo 3)")

    if carrito:
        mostrar_perfil_de_generos(
            pd.Series(carrito),
            "📊 Tu perfil de gustos según lo que puntuaste"
        )

    if st.button("🔍 Ver recomendaciones", type="primary",
                 width='stretch', key="btn_nuevo"):
        if len(carrito) < 3:
            st.warning("Puntuá al menos 3 películas para que podamos recomendarte algo.")
        else:
            # sin spinner: con el modelo pre-entrenado esto es instantáneo
            semillas = dict(carrito)
            recomendaciones = recomendar_para_usuario_nuevo(
                similitud_df, movies, semillas, n=n_recomendaciones
            )
            # guardamos el resultado en session_state para que sobreviva a
            # los reruns (cada interacción con cualquier widget re-ejecuta
            # todo el script; sin esto, las recomendaciones desaparecerían)
            st.session_state['resultado_nuevo'] = {
                'recomendaciones': recomendaciones,
                'subtitulo': (
                    "Calculadas **solo con el modelo de contenido** (similitud "
                    "de géneros con tus películas semilla). El modelo "
                    "colaborativo necesita historial dentro del dataset, así "
                    "que acá no participa."
                ),
                'explicaciones': explicaciones_para(
                    recomendaciones, pd.Series(semillas), por_peso=True
                ),
            }

    # se muestra desde session_state: persiste hasta el próximo cálculo
    if 'resultado_nuevo' in st.session_state:
        resultado = st.session_state['resultado_nuevo']
        mostrar_recomendaciones(
            resultado['recomendaciones'], resultado['subtitulo'],
            explicaciones=resultado['explicaciones'],
        )

with tab_existente:
    # ── Usuario existente: flujo híbrido normal ──────────────
    userId = st.number_input(
        label="ID de usuario (1 a 943)",
        min_value=1, max_value=943, value=1, step=1
    )

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
                hide_index=True, width='stretch'
            )

        mostrar_perfil_de_generos(
            vistas_usuario.set_index('movieId')['rating'],
            f"📊 Perfil de gustos por género del usuario {userId}"
        )

    if st.button("🔍 Ver recomendaciones", type="primary",
                 width='stretch', key="btn_existente"):
        # sin spinner: con el modelo pre-entrenado esto es instantáneo
        recomendaciones = recomendar(userId=userId, alpha=alpha, n=n_recomendaciones)

        if recomendaciones is None:
            st.error(f"El usuario {userId} no existe en el dataset.")
        else:
            historial = (
                ratings[ratings['userId'] == userId]
                .set_index('movieId')['rating']
            )
            # persistimos el resultado para que no desaparezca cuando otro
            # widget (el slider de alpha, el sidebar) provoque un rerun
            st.session_state['resultado_existente'] = {
                'recomendaciones': recomendaciones,
                'subtitulo': (
                    f"Para el usuario **{userId}**, con alpha=**{alpha:.1f}** "
                    f"({int(alpha*100)}% colaborativo · "
                    f"{int((1-alpha)*100)}% contenido)"
                ),
                'explicaciones': explicaciones_para(recomendaciones, historial),
            }

    # se muestra desde session_state: persiste hasta el próximo cálculo
    if 'resultado_existente' in st.session_state:
        resultado = st.session_state['resultado_existente']
        mostrar_recomendaciones(
            resultado['recomendaciones'], resultado['subtitulo'],
            explicaciones=resultado['explicaciones'],
        )

st.divider()
st.caption(
    "Proyecto de portafolio · Licenciatura en Ciencias de Datos · UBA · "
    "Dataset: [MovieLens 100K](https://grouplens.org/datasets/movielens/100k/)"
)
