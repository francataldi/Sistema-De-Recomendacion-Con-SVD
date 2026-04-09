# 🎬 Sistema de Recomendación Híbrido con SVD

> Proyecto de portafolio · Licenciatura en Ciencias de Datos · UBA · 3er año

**[▶ Ver demo en vivo](https://sistema-de-recomendacion-con-svd.streamlit.app/)**

---

## ¿Qué hace este proyecto?

Dado el historial de ratings de un usuario, el sistema predice qué películas le van a gustar — incluyendo películas que nunca vio y para las que no existe ningún dato previo.

El desafío central: la mayoría de los usuarios no vio la mayoría de las películas. El **93.7% de los datos posibles no existen**. Ese vacío es el verdadero problema a resolver, y es exactamente para lo que SVD fue diseñado.

---

## Demo

El sistema recibe un ID de usuario y devuelve las 10 películas más recomendadas, con un score visual. El slider de alpha permite experimentar con el balance entre los dos modelos en tiempo real.

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

Cada enfoque resuelve el punto débil del otro. La combinación es:

```
score_final = α × score_colaborativo + (1 - α) × score_contenido
```

El valor óptimo de α se encontró evaluando el RMSE sobre un conjunto de prueba separado.

---

## Por qué SVD

La matriz usuario × película tiene 943 filas y 1.682 columnas — pero el **93.7% de las celdas están vacías**. Una matriz así se llama *sparse* (dispersa) y no se puede usar directamente para recomendar.

SVD descompone esa matriz en tres matrices más pequeñas:

```
M ≈ U × Σ × Vᵀ

U   →  943  × k   (perfil de cada usuario en k factores latentes)
Σ   →  k    × k   (importancia de cada factor)
Vᵀ  →  k    × 1682  (perfil de cada película en los mismos factores)
```

Los **factores latentes** son patrones que el algoritmo descubre solo — nadie los nombra ni los define. Al reconstruir la matriz desde estas tres, los huecos quedan "rellenados" con ratings estimados. Esas estimaciones son las recomendaciones.

---

## Resultados

| Experimento | Resultado |
|---|---|
| Sparsity de la matriz | 93.7% |
| Factores latentes usados (k) | 50 |
| RMSE colaborativo (k=50, sobre entrenamiento) | 0.728 |
| Mejor alpha encontrado | 0.7 |
| RMSE del modelo híbrido (sobre conjunto de prueba) | 1.188 |

> El RMSE sobre entrenamiento (0.728) mide qué tan bien SVD reconstruye datos que ya vio. El RMSE sobre prueba (1.188) mide la capacidad real de predicción sobre datos nuevos — ese es el número honesto.

---

## Estructura del repositorio

```
Recomendador-SVD/
│
├── Data/
│   └── ml-100k/
│       ├── u.data     ← ratings de usuarios
│       └── u.item     ← metadata de películas
│
├── notebooks/
│   └── main.ipynb     ← desarrollo completo paso a paso
│
├── app.py             ← interfaz Streamlit
├── requirements.txt
└── README.md
```

---

## Correr localmente

```bash
git clone https://github.com/francataldi/Sistema-De-Recomendacion-Con-SVD.git
cd Sistema-De-Recomendacion-Con-SVD

pip install -r requirements.txt

streamlit run app.py
# El modelo se entrena automáticamente al iniciar la app (~15 segundos)
```

---

## Stack

| Herramienta | Uso |
|---|---|
| Python 3.x | Lenguaje base |
| pandas | Carga y manipulación de datos |
| numpy | Álgebra lineal, SVD |
| scipy | `svds` — SVD eficiente para matrices sparse |
| scikit-learn | Similitud coseno, métricas |
| Streamlit | Interfaz web y deploy |

---

## Referencias

- [MovieLens 100K — GroupLens](https://grouplens.org/datasets/movielens/100k/)
- [Matrix Factorization Techniques for Recommender Systems — Koren, Bell & Volinsky (2009)](https://datajobs.com/data-science-repo/Recommender-Systems-%5BNetflix%5D.pdf) — el paper del Netflix Prize que popularizó SVD en recomendación
- [Streamlit Docs](https://docs.streamlit.io/)
