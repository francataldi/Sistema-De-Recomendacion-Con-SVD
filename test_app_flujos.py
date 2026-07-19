# -*- coding: utf-8 -*-
"""
Tests de flujo de la app de Streamlit (app.py) con AppTest.

Cubre los dos modos de la interfaz:
  - Tab "👤 Usuario del dataset": recomendaciones híbridas + persistencia.
  - Tab "✨ Usuario nuevo": buscador de semillas + carrito editable.

Se corren con:  pytest test_app_flujos.py -v

Nota: en los tests desactivamos la API key de TMDb (at.secrets vacío) para
que los pósters no se busquen — así los tests son deterministas y no
dependen de la red ni de tener una key configurada. La lógica de
recomendación es la misma con o sin pósters.
"""
import os
import sys

import pytest
from streamlit.testing.v1 import AppTest

RUTA = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, RUTA)
os.chdir(RUTA)
APP = os.path.join(RUTA, "app.py")


# ── Helpers ───────────────────────────────────────────────────

def app_fresca():
    """Instancia y corre la app desde cero, sin API key de TMDb."""
    at = AppTest.from_file(APP, default_timeout=90)
    at.secrets["TMDB_API_KEY"] = ""   # pósters desactivados: tests sin red
    at.run()
    assert not at.exception, f"La app lanzó una excepción al cargar: {at.exception}"
    return at


def slider_por_key(at, key):
    for s in at.select_slider:
        if s.key == key:
            return s
    return None


def keys_de_tarjetas(at):
    """Keys de los select_slider de las tarjetas puntuables mostradas."""
    return [s.key for s in at.select_slider if str(s.key).startswith("sem_punt_")]


def mid_de_key(key):
    return int(str(key).split("_")[-1])


def texto_markdown(at):
    return " ".join(str(m.value) for m in at.markdown)


def carrito(at):
    return at.session_state["carrito_semillas"]


# ── Carga base ────────────────────────────────────────────────

def test_carga_inicial_sin_excepcion():
    at = app_fresca()
    # ambos tabs se renderizan siempre (Streamlit oculta el inactivo)
    assert "🎬 Recomendador de Películas" in texto_markdown(at) or at.title


# ── Tab "Usuario del dataset" (cobertura vieja, no debe romperse) ──

def test_usuario_existente_genera_recomendaciones():
    at = app_fresca()
    at.button(key="btn_existente").click().run()
    assert not at.exception
    subheaders = [s.value for s in at.subheader]
    assert any("recomendaciones" in s for s in subheaders)
    # explicación del híbrido: "Similar a ..."
    explicaciones = [c.value for c in at.caption if "Similar a" in str(c.value)]
    assert explicaciones, "Faltan las explicaciones 'Similar a ...' del híbrido"


def test_usuario_existente_persiste_tras_rerun():
    at = app_fresca()
    at.button(key="btn_existente").click().run()
    assert any("recomendaciones" in s.value for s in at.subheader)
    # mover el slider de alpha dispara un rerun; las recomendaciones deben
    # seguir en pantalla (viven en session_state)
    at.slider[0].set_value(0.5).run()
    assert not at.exception
    assert any("recomendaciones" in s.value for s in at.subheader), \
        "Las recomendaciones desaparecieron tras un rerun"


# ── Tab "Usuario nuevo": buscador + carrito ───────────────────

def test_nuevo_muestra_populares_por_defecto():
    at = app_fresca()
    # sin buscar nada, se muestran las populares como tarjetas puntuables
    tarjetas = keys_de_tarjetas(at)
    assert len(tarjetas) == 8, f"Se esperaban 8 populares, hay {len(tarjetas)}"


def test_nuevo_puntuar_popular_agrega_al_carrito():
    at = app_fresca()
    key = keys_de_tarjetas(at)[0]
    mid = mid_de_key(key)
    slider_por_key(at, key).set_value("5").run()
    assert not at.exception
    assert carrito(at).get(mid) == 5


def test_nuevo_busqueda_conserva_carrito_y_encuentra():
    at = app_fresca()
    # puntúo una popular
    key = keys_de_tarjetas(at)[0]
    mid = mid_de_key(key)
    slider_por_key(at, key).set_value("4").run()
    assert carrito(at).get(mid) == 4

    # busco "Godfather": debe aparecer "Godfather, The (1972)"
    at.text_input(key="busqueda_semillas").set_value("Godfather").run()
    assert not at.exception
    assert "Godfather, The" in texto_markdown(at)
    # y la puntuación anterior NO se perdió al cambiar de vista
    assert carrito(at).get(mid) == 4, "Se perdió la puntuación al buscar"


def test_nuevo_puntuar_desde_busqueda_suma_al_carrito():
    at = app_fresca()
    # puntúo una popular primero
    key_pop = keys_de_tarjetas(at)[0]
    mid_pop = mid_de_key(key_pop)
    slider_por_key(at, key_pop).set_value("5").run()

    # busco Godfather y puntúo el resultado (movieId 127)
    at.text_input(key="busqueda_semillas").set_value("Godfather").run()
    slider_por_key(at, "sem_punt_127").set_value("4").run()
    assert not at.exception
    # el carrito tiene AMBAS (la de búsqueda no reemplazó a la popular)
    assert carrito(at).get(mid_pop) == 5
    assert carrito(at).get(127) == 4


def test_nuevo_sacar_del_carrito_baja_el_contador():
    at = app_fresca()
    key = keys_de_tarjetas(at)[0]
    mid = mid_de_key(key)
    slider_por_key(at, key).set_value("5").run()
    assert len(carrito(at)) == 1

    # click en "❌ Sacar" del ítem del carrito
    botones = [b for b in at.button if b.key == f"quitar_{mid}"]
    assert botones, f"No se encontró el botón Sacar (quitar_{mid})"
    botones[0].click().run()
    assert not at.exception
    assert mid not in carrito(at)
    assert len(carrito(at)) == 0


def test_nuevo_menos_de_3_muestra_warning():
    at = app_fresca()
    # puntúo solo 2 películas
    tarjetas = keys_de_tarjetas(at)
    slider_por_key(at, tarjetas[0]).set_value("5").run()
    slider_por_key(at, tarjetas[1]).set_value("4").run()
    assert len(carrito(at)) == 2

    at.button(key="btn_nuevo").click().run()
    assert not at.exception
    warnings = [w.value for w in at.warning]
    assert any("al menos 3" in str(w) for w in warnings), "Falta el warning de mínimo 3"
    # no se generó ningún resultado
    assert "resultado_nuevo" not in at.session_state


def test_nuevo_3_o_mas_genera_recomendaciones():
    at = app_fresca()
    tarjetas = keys_de_tarjetas(at)
    for key, val in zip(tarjetas[:3], ["5", "4", "3"]):
        slider_por_key(at, key).set_value(val).run()
    assert len(carrito(at)) == 3

    at.button(key="btn_nuevo").click().run()
    assert not at.exception
    assert any("recomendaciones" in s.value for s in at.subheader)
    # explicación de cold start: "Tu semilla que más pesó ..."
    explicaciones = [c.value for c in at.caption if "pesó" in str(c.value)]
    assert explicaciones, "Faltan las explicaciones 'Tu semilla que más pesó ...'"


def test_nuevo_busqueda_inexistente_muestra_info():
    at = app_fresca()
    # "Matrix" (1999) no está en MovieLens 100k (catálogo hasta abril 1998)
    at.text_input(key="busqueda_semillas").set_value("Matrix").run()
    assert not at.exception
    infos = [i.value for i in at.info]
    assert any("No encontré" in str(i) for i in infos), \
        "Debería mostrar el mensaje de 'no encontré' sin romperse"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
