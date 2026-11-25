"""Elementos compartidos por las páginas del dashboard: datos, paleta y estilo de gráficas."""
from pathlib import Path
from urllib.parse import quote

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from iconos import LUCIDE

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
MODELS = ROOT / "models"

# Paleta (validada para daltonismo en pares: azul, naranja, aqua + gris neutro
# para el segmento inactivo, que además se dibuja con menor énfasis).
ORDEN = ["Campeones", "Leales en riesgo", "Nuevos / ocasionales", "Inactivos"]
COLORES = {"Campeones": "#2a78d6", "Leales en riesgo": "#eb6834",
           "Nuevos / ocasionales": "#1baf7a", "Inactivos": "#b5b4ae"}
SIMBOLOS = {"Campeones": "circle", "Leales en riesgo": "diamond",
            "Nuevos / ocasionales": "square", "Inactivos": "triangle-up"}
# Icono de Lucide de cada segmento y color del icono (el gris de Inactivos se oscurece para que se lea)
ICONOS = {"Campeones": "trophy", "Leales en riesgo": "triangle-alert", "Nuevos / ocasionales": "sprout",
          "Inactivos": "moon"}
COLOR_ICONO = {**COLORES, "Inactivos": "#9a9993"}
AZUL = "#2a78d6"; GRIS = "#9a9993"
TEXTO = "#0b0b0b"; TEXTO_2 = "#52514e"; GRID = "#e6e5e0"

ETIQUETAS = {"Recencia": "Recencia (días)", "Frecuencia": "Frecuencia (compras)",
             "Monetario": "Monetario (£)", "TicketPromedio": "Ticket promedio (£)",
             "Productos": "Productos distintos", "Antiguedad": "Antigüedad (días)",
             "TasaDevolucion": "Tasa de devolución (%)"}
LOG_VARS = ("Frecuencia", "Monetario", "TicketPromedio", "Productos")

EUROPA = {"Austria", "Belgium", "Channel Islands", "Cyprus", "Czech Republic", "Denmark", "Finland", "France",
          "Germany", "Greece", "Iceland", "Ireland", "Italy", "Lithuania", "Malta", "Netherlands", "Norway",
          "Poland", "Portugal", "Spain", "Sweden", "Switzerland"}
DIAS = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
MESES = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]


def icono(nombre, tam="1em", color="currentColor", separacion="0"):
    """SVG en línea de un icono de Lucide, listo para st.markdown(..., unsafe_allow_html=True)."""
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{tam}" height="{tam}" viewBox="0 0 24 24" fill="none" '
            f'stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" '
            f'style="color:{color};vertical-align:-0.125em;margin-right:{separacion}" aria-hidden="true">'
            f'{LUCIDE[nombre]}</svg>')


def encabezado(texto, nombre, nivel=2, color="currentColor"):
    """Título (h1, h2 o h3) con un icono de Lucide a la izquierda."""
    st.markdown(f"<h{nivel}>{icono(nombre, '0.85em', color, '0.4em')}{texto}</h{nivel}>", unsafe_allow_html=True)


def css_icono(selector, nombre, tam="1.1rem"):
    """Regla CSS que dibuja un icono de Lucide antes de un elemento que no acepta HTML (menú, botones).

    Usa el SVG como máscara, así el icono toma el color del texto del elemento."""
    url = "data:image/svg+xml," + quote(icono(nombre, "24", "black"))
    return (f"{selector}::before {{content:'';display:inline-block;flex-shrink:0;width:{tam};height:{tam};"
            f"margin-right:0.5rem;background-color:currentColor;"
            f"-webkit-mask:url(\"{url}\") center/contain no-repeat;mask:url(\"{url}\") center/contain no-repeat;}}")


def estilo(fig, alto=380):
    fig.update_layout(template="plotly_white", height=alto, margin=dict(l=10, r=10, t=75, b=10),
                      font=dict(color=TEXTO_2, size=13), title_font=dict(color=TEXTO, size=15),
                      title=dict(y=0.98, yanchor="top", x=0, xanchor="left"),
                      legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, title=None),
                      hoverlabel=dict(bgcolor="white", font_color=TEXTO))
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def rango_log(serie):
    """Rango del eje log: desde el percentil 0.5 (mínimo 1) para que valores casi nulos no aplasten la gráfica."""
    bajo = max(serie.quantile(0.005), 1)
    return [np.log10(bajo) - 0.1, np.log10(serie.max()) + 0.1]


def region(pais):
    return np.select([pais == "United Kingdom", pais.isin(EUROPA)], ["Reino Unido", "Europa"],
                     default="Resto del mundo")


@st.cache_data
def cargar_clientes():
    df = pd.read_csv(DATA / "clientes_segmentados.csv")
    df["TasaDevolucion"] = df["TasaDevolucion"] * 100  # a porcentaje
    df["Segmento"] = pd.Categorical(df["Segmento"], categories=ORDEN, ordered=True)
    return df


@st.cache_data
def cargar_ventas():
    v = pd.read_csv(DATA / "ventas_limpias.csv.gz", dtype={"InvoiceNo": str, "StockCode": str})
    v["InvoiceDate"] = pd.to_datetime(v["InvoiceDate"])
    v["Region"] = region(v["Country"])
    v["Mes"] = v["InvoiceDate"].dt.to_period("M").dt.to_timestamp()
    v["DiaSemana"] = v["InvoiceDate"].dt.dayofweek
    v["Hora"] = v["InvoiceDate"].dt.hour
    return v


@st.cache_data
def cargar_evaluacion():
    return pd.read_csv(DATA / "evaluacion_k.csv")


@st.cache_data
def cargar_bitacora():
    return pd.read_csv(DATA / "bitacora_limpieza.csv")


@st.cache_resource
def cargar_modelo():
    return joblib.load(MODELS / "kmeans_rfm.joblib")


def centroides():
    """Centroides del modelo en unidades originales (se deshace el escalamiento y el logaritmo)."""
    m = cargar_modelo()
    orig = np.expm1(m["scaler"].inverse_transform(m["kmeans"].cluster_centers_))
    c = pd.DataFrame(orig, columns=["Recencia (días)", "Frecuencia (compras)", "Monetario (£)"])
    c.insert(0, "Clúster", range(len(c)))
    c.insert(0, "Segmento", [m["nombres"][i] for i in range(len(c))])
    return c.set_index("Segmento").loc[ORDEN]
