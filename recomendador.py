"""
Lógica de recomendación compartida entre el notebook (main.ipynb) y la app
de Streamlit (app.py).

Toda la lógica de scoring vive acá para no tener dos copias que puedan
desincronizarse. Las funciones reciben explícitamente el modelo y los datos
(no usan variables globales), así se pueden probar sueltas desde el notebook.
"""

import re

import numpy as np
import pandas as pd

# Artículos que MovieLens mueve al final del título ("Matrix, The").
_ARTICULOS = r'The|A|An|La|Le|Les|Los|El|Il|Das|Der|Die'


def titulo_normalizado(titulo):
    """
    Devuelve la forma "natural" de un título de MovieLens, lista para
    buscar o mostrar:

    - saca el año final "(1994)",
    - saca un título alternativo entre paréntesis al final
      (ej. "Like Water For Chocolate (Como agua para chocolate)"),
    - mueve el artículo que MovieLens pone al final al principio
      ("Matrix, The" -> "The Matrix").

    Es utilidad COMPARTIDA: la usa tanto buscar_peliculas() (para tolerar
    el artículo al buscar) como limpiar_titulo() en app.py (para armar la
    query de TMDb). Vive acá, junto a la búsqueda, para no duplicarla.
    """
    limpio = re.sub(r'\s*\(\d{4}\)\s*$', '', titulo)
    limpio = re.sub(r'\s*\([^)]*\)\s*$', '', limpio)
    m = re.match(rf'^(?P<resto>.+),\s+(?P<articulo>{_ARTICULOS})$', limpio)
    if m:
        limpio = f"{m.group('articulo')} {m.group('resto')}"
    return limpio.strip()


def buscar_peliculas(movies, query, n=12):
    """
    Busca películas por título, tolerando el formato de MovieLens donde el
    artículo va al final (ej. "Matrix, The (1999)" debe encontrarse
    buscando "Matrix" o "The Matrix"). Devuelve hasta n resultados
    ordenados por popularidad (si movies trae esa info en una columna
    'cantidad') o alfabéticamente si no.

    No es fuzzy matching: normaliza el artículo y hace un contains
    case-insensitive contra el título tal cual Y contra su forma
    normalizada, así "Matrix" y "The Matrix" encuentran lo mismo.
    """
    q = str(query).strip().lower()
    if not q:
        return movies.head(0)

    titulos = movies['title']
    normalizados = titulos.map(titulo_normalizado)
    mask = (
        titulos.str.lower().str.contains(q, regex=False)
        | normalizados.str.lower().str.contains(q, regex=False)
    )
    resultado = movies[mask]

    # popularidad si el dataframe la trae; si no, orden alfabético estable
    if 'cantidad' in resultado.columns:
        resultado = resultado.sort_values('cantidad', ascending=False)
    else:
        resultado = resultado.sort_values('title')
    return resultado.head(n)


def score_colaborativo(modelo_svd, userId, candidatas):
    """
    Rating predicho (escala real 1-5) por el modelo SVD de scikit-surprise
    para cada película candidata.

    - modelo_svd: modelo SVD de surprise ya entrenado
    - userId: id del usuario (el "raw id" del dataset)
    - candidatas: lista de movieIds a puntuar

    Devuelve una Series indexada por movieId.
    """
    return pd.Series(
        {m: modelo_svd.predict(userId, m).est for m in candidatas}
    )


def score_contenido(similitud_df, ratings_usuario, candidatas):
    """
    Score de contenido para cada película candidata: similitud de género
    con las películas que el usuario ya vio, ponderada por el rating que
    les dio. NO está en escala de rating — solo sirve para ordenar.

    - similitud_df: DataFrame movieId x movieId con similitud coseno
    - ratings_usuario: Series indexada por movieId con los ratings del
      usuario (su historial, o sus películas "semilla" si es nuevo)
    - candidatas: lista de movieIds a puntuar

    Devuelve una Series indexada por movieId.
    """
    vistas = ratings_usuario[ratings_usuario.index.isin(similitud_df.index)]
    sims = similitud_df.loc[candidatas, vistas.index].values
    scores = sims @ vistas.values / (vistas.values.sum() + 1e-8)
    return pd.Series(scores, index=candidatas)


def _normalizar_01(s):
    """Lleva una Series a [0, 1]. Transformación monótona: no cambia el orden."""
    return (s - s.min()) / (s.max() - s.min() + 1e-8)


def recomendar_hibrido(modelo_svd, similitud_df, movies, ratings_usuario,
                       userId, alpha=0.7, n=10):
    """
    Top-n de películas para un usuario CON historial, combinando ambos
    modelos: score = alpha * colaborativo + (1-alpha) * contenido, con los
    dos scores normalizados a [0, 1]. El score final ordena, no predice
    ratings.

    - ratings_usuario: Series indexada por movieId con el historial del usuario

    Devuelve un DataFrame con columnas ['title', 'score_final'], o None si
    el usuario no tiene historial.
    """
    if ratings_usuario is None or ratings_usuario.empty:
        return None

    candidatas = [m for m in similitud_df.index if m not in ratings_usuario.index]

    colab = score_colaborativo(modelo_svd, userId, candidatas)
    cont = score_contenido(similitud_df, ratings_usuario, candidatas)

    score_final = alpha * _normalizar_01(colab) + (1 - alpha) * _normalizar_01(cont)
    top_n = score_final.sort_values(ascending=False).head(n)

    return (
        pd.DataFrame({'movieId': top_n.index, 'score_final': top_n.values})
        .merge(movies[['movieId', 'title']], on='movieId')
        [['movieId', 'title', 'score_final']]
    )


def peliculas_populares(ratings, movies, n=20, min_ratings=100):
    """
    Las n películas más populares del dataset: entre las que tienen al menos
    min_ratings ratings, las de mejor rating promedio. Se usan como lista de
    "semillas" para que un usuario nuevo puntúe algo conocido (cold start).

    Devuelve un DataFrame con ['movieId', 'title', 'rating_promedio', 'cantidad'].
    """
    stats = (
        ratings.groupby('movieId')['rating']
        .agg(rating_promedio='mean', cantidad='count')
        .reset_index()
    )
    populares = (
        stats[stats['cantidad'] >= min_ratings]
        .sort_values('rating_promedio', ascending=False)
        .head(n)
        .merge(movies[['movieId', 'title']], on='movieId')
    )
    return populares[['movieId', 'title', 'rating_promedio', 'cantidad']]


def recomendar_para_usuario_nuevo(similitud_df, movies, semillas, n=10):
    """
    Recomendaciones para un usuario que NO existe en el dataset (cold start),
    usando SOLO el modelo de contenido: el colaborativo no tiene factores
    aprendidos para un usuario que nunca vio en el entrenamiento, así que no
    puede aportar nada útil acá.

    - semillas: Series (o dict) movieId -> rating (1-5) con las pocas
      películas que el usuario nuevo puntuó de arranque.

    Devuelve un DataFrame con ['title', 'score_final'] (score de contenido
    normalizado a [0, 1]), o None si no hay semillas.
    """
    if semillas is None or len(semillas) == 0:
        return None
    semillas = pd.Series(semillas)

    candidatas = [m for m in similitud_df.index if m not in semillas.index]
    cont = score_contenido(similitud_df, semillas, candidatas)

    top_n = _normalizar_01(cont).sort_values(ascending=False).head(n)

    return (
        pd.DataFrame({'movieId': top_n.index, 'score_final': top_n.values})
        .merge(movies[['movieId', 'title']], on='movieId')
        [['movieId', 'title', 'score_final']]
    )


def explicar_recomendacion(similitud_df, ratings_usuario, movieId,
                           por_peso=False):
    """
    Explica una recomendación: de las películas que el usuario ya puntuó,
    ¿cuál es la que justifica esta recomendación?

    - por_peso=False (híbrido): devuelve la película vista MÁS SIMILAR en
      género a la recomendada.
    - por_peso=True (usuario nuevo): devuelve la semilla que MÁS PESÓ en el
      score de contenido, es decir la que maximiza similitud x rating —
      exactamente el término más grande de la suma que calcula
      score_contenido().

    Devuelve (movieId_similar, similitud, rating_dado), o None si no hay
    con qué explicar.
    """
    vistas = ratings_usuario[ratings_usuario.index.isin(similitud_df.index)]
    if len(vistas) == 0 or movieId not in similitud_df.index:
        return None

    sims = similitud_df.loc[movieId, vistas.index]
    if por_peso:
        mejor = (sims * vistas).idxmax()
    else:
        mejor = sims.idxmax()
    return mejor, float(sims.loc[mejor]), float(vistas.loc[mejor])
