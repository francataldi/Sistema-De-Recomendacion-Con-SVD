# 🎬 Sistema de Recomendación Híbrido con SVD

> Proyecto de portafolio · Licenciatura en Ciencias de Datos · UBA · 3er año

<!-- TODO: actualizar tras el deploy con el link real de Streamlit Cloud -->
**Demo en vivo:** aún no desplegada — pendiente de deploy en Streamlit
Community Cloud.

---

## ¿Qué hace este proyecto?

Dado el historial de ratings de un usuario, el sistema predice qué películas le van a gustar — incluyendo películas que nunca vio y para las que no existe ningún dato previo.

El desafío central: la mayoría de los usuarios no vio la mayoría de las películas. El **93.7% de los datos posibles no existen**. Ese vacío es el verdadero problema a resolver, y es exactamente para lo que SVD fue diseñado.

---

## Interfaz

La app de Streamlit está organizada en dos pestañas según quién la use:

- **👤 Usuario del dataset:** ingresás un ID (1 a 943) y recibís tu Top-N híbrido. Antes de recomendar podés desplegar tu historial de películas vistas y un gráfico de tu perfil de gustos por género.
- **✨ Usuario nuevo:** si el sistema no te conoce, puntuás al menos 3 películas populares ("semillas") y con eso se arma tu perfil de gustos para recomendarte por contenido (ver [Cold start](#modelo-híbrido)).

Cada recomendación se muestra como una **tarjeta** con:

- **Póster de la película** (vía la API de TMDb). Los pósters son *opcionales*: si no hay API key configurada, la app avisa y sigue funcionando en modo texto, con un placeholder en lugar de la imagen.
- **Tags de género de colores**, con un color fijo por género para reconocer patrones de un vistazo.
- **Explicación del "por qué"**: para usuarios del dataset, la película de tu historial más parecida a la recomendada (ej. *"Similar a Cinema Paradiso, que calificaste con 5★"*); para usuarios nuevos, la semilla que más pesó en esa recomendación.
- **Score visual** con una barra de progreso.

En el **sidebar** están los controles de configuración: cantidad de recomendaciones y el slider de alpha, que permite experimentar en vivo con el balance entre los dos modelos.

### Pósters vía TMDb — configuración de la API key

Los pósters usan la [API de TMDb](https://www.themoviedb.org/settings/api). La key **nunca se sube al repo** ni se escribe en el código: se lee exclusivamente de `st.secrets["TMDB_API_KEY"]`. Configurala según dónde corras la app:

- **Local:** creá el archivo `.streamlit/secrets.toml` (ya está en `.gitignore`) con:
  ```toml
  TMDB_API_KEY = "tu_key_aca"
  ```
- **Streamlit Community Cloud:** cargala en el panel *Settings → Secrets* de la app, con el mismo nombre `TMDB_API_KEY`.

Sin key configurada, la app funciona igual: solo se muestra en modo texto, sin pósters.

---

## Dataset

**MovieLens 100K** — GroupLens Research, Universidad de Minnesota.

| Característica | Valor |
|---|---|
| Total de ratings | 100.000 |
| Usuarios únicos | 943 |
| Películas únicas | 1.682 |
| Escala | 1 a 5 estrellas |
| Período | Septiembre 1997 – Abril 1998 |

---

## Cómo funciona

El sistema combina dos enfoques complementarios:

### Filtrado Colaborativo (SVD)

Encuentra patrones entre usuarios con gustos similares. Si Franco y Martín coincidieron en 5 estrellas para Toy Story e Interstellar, y Franco también amó The Matrix, el sistema recomienda The Matrix a Martín — sin saber nada sobre la película en sí.

**Punto débil:** no funciona con usuarios nuevos sin historial (*cold start*).

### Filtrado Basado en Contenido

Cada película está representada como un vector de 19 géneros binarios. La similitud entre dos películas se mide con el coseno del ángulo entre sus vectores — álgebra lineal pura.

**Punto débil:** tiende a recomendar siempre lo mismo (*sobre-especialización*).

### Modelo Híbrido

La combinación es:

```
score_final = α × score_colaborativo + (1 - α) × score_contenido
```

El valor óptimo de α se encontró maximizando NDCG@10 sobre un conjunto de prueba separado.

**Qué resuelve el híbrido y qué no** (esto es importante decirlo con precisión):

- **Cold start de películas nuevas**: lo resuelve el modelo de contenido, porque solo necesita los géneros de la película — no hace falta que nadie la haya rateado todavía.
- **Cold start de usuarios nuevos**: *no* se resuelve automáticamente por ser híbrido. Ambos scores necesitan historial del usuario: el colaborativo necesita que el usuario haya estado en el entrenamiento, y el de contenido necesita saber qué películas le gustaron. Para usuarios nuevos, la app implementa un flujo de **películas semilla**: el usuario puntúa al menos 3 películas populares y con eso se calculan recomendaciones usando solo el modelo de contenido (el colaborativo no tiene factores aprendidos para un usuario que nunca vio).
- **Sobre-especialización del contenido**: en el ranking Top-N, la mezcla 70/30 le gana a cada modelo por separado, aportando diversidad real.

---

## Por qué SVD (y cuál SVD)

La matriz usuario × película tiene 943 filas y 1.682 columnas — pero el **93.7% de las celdas están vacías**. Una matriz así se llama *sparse* (dispersa) y no se puede usar directamente para recomendar.

La idea de la factorización es representar a cada usuario y a cada película con un vector de `k` **factores latentes** — patrones que el algoritmo descubre solo, nadie los nombra ni los define — y predecir cada rating como:

```
rating estimado = media global + sesgo del usuario + sesgo de la película + (usuario · película)
```

Este proyecto usa el algoritmo `SVD` de `scikit-surprise` (el "Funk SVD" del Netflix Prize), que ajusta esos factores por descenso de gradiente regularizado **solo sobre los ratings observados**.

**Nota pedagógica:** la primera versión del proyecto usaba `scipy.sparse.linalg.svds` sobre la matriz densa, rellenando las celdas vacías con la media de cada usuario. Eso es un error metodológico (el modelo termina aprendiendo a reproducir el relleno artificial, no los gustos reales) y el notebook lo documenta paso a paso como contraejemplo, junto con la corrección.

---

## Resultados

El proyecto evalúa dos tareas distintas, cada una con su métrica correcta:

**Tarea 1 — Predicción de rating** (¿cuántas estrellas le pondría el usuario a esta película?), evaluada con RMSE sobre un 20% de test que el modelo nunca vio:

| Modelo | RMSE test |
|---|---|
| Baseline: media global | 1.130 |
| Baseline: media + sesgo de usuario + sesgo de película | 0.965 |
| **SVD scikit-surprise (k=10 óptimo)** | **0.933** |

**Tarea 2 — Ranking Top-10** (¿qué 10 películas le muestro primero?), evaluada con métricas de ranking (relevante = rating ≥ 4 en test):

| Configuración | Precision@10 | Recall@10 | NDCG@10 |
|---|---|---|---|
| Solo contenido (alpha=0) | 0.018 | 0.018 | 0.022 |
| Solo colaborativo (alpha=1) | 0.066 | 0.045 | 0.072 |
| **Híbrido (alpha=0.7 óptimo)** | **0.072** | **0.057** | **0.087** |

> Para predecir ratings el colaborativo hace casi todo el trabajo; para rankear el Top-10, la mezcla 70% colaborativo / 30% contenido le gana a ambos modelos por separado. Los detalles y el porqué de cada decisión están documentados paso a paso en el notebook.

---

## Estructura del repositorio

```
Recomendador-SVD/
│
├── Data/                        ← NO está en el repo (se descarga aparte)
│   └── ml-100k/
│
├── modelo/
│   └── modelo_hibrido.pkl       ← modelo entrenado serializado (lo genera el notebook)
│
├── main.ipynb                   ← desarrollo completo paso a paso
├── recomendador.py              ← lógica de recomendación compartida (notebook + app)
├── app.py                       ← interfaz Streamlit
├── requirements.txt
└── README.md
```

---

## Correr localmente

```bash
git clone https://github.com/francataldi/Recomendador-SVD.git
cd Recomendador-SVD

pip install -r requirements.txt

# Descargar el dataset desde https://grouplens.org/datasets/movielens/100k/
# y colocarlo en Data/ml-100k/

# (Opcional) Ejecutar el notebook para regenerar el modelo.
# El repo ya incluye modelo/modelo_hibrido.pkl, así que la app
# funciona directo; el notebook está en la raíz del repo.
jupyter notebook main.ipynb

# Lanzar la app
streamlit run app.py
```

---

## Stack

| Herramienta | Uso |
|---|---|
| Python 3.11 | Lenguaje base |
| pandas | Carga y manipulación de datos |
| numpy | Álgebra lineal |
| scikit-surprise | SVD por gradiente descendente (modelo colaborativo) |
| scipy | `svds` — usado en la versión inicial (contraejemplo pedagógico) |
| scikit-learn | Similitud coseno, split train/test |
| matplotlib / seaborn | Gráficos del notebook |
| Streamlit | Interfaz web y deploy |
| requests | Llamadas a la API de TMDb (pósters) |
| TMDb API | Pósters de las películas (opcional) |

---

## Referencias

- [MovieLens 100K — GroupLens](https://grouplens.org/datasets/movielens/100k/)
- [Matrix Factorization Techniques for Recommender Systems — Koren, Bell & Volinsky (2009)](https://datajobs.com/data-science-repo/Recommender-Systems-%5BNetflix%5D.pdf) — el paper del Netflix Prize que popularizó SVD en recomendación
- [Streamlit Docs](https://docs.streamlit.io/)
