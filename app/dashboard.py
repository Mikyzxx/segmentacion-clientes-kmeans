"""Dashboard interactivo — Segmentación de clientes con K-Means (RFM).

Dos páginas:
  1. Análisis general: indicadores, estadísticas y visualizaciones del dataset limpio.
  2. Segmentación K-Means: agrupamientos, características de los clústeres y resultados del modelo.

Ejecutar desde la raíz del repositorio:
    streamlit run app/dashboard.py
"""
from pathlib import Path

import streamlit as st

from comun import AZUL, css_icono, encabezado

st.set_page_config(page_title="Segmentación de clientes K-Means",
                   page_icon=str(Path(__file__).parent / "favicon.svg"), layout="wide")

pagina = st.navigation([
    st.Page("paginas/analisis_general.py", title="1. Análisis general", default=True),
    st.Page("paginas/segmentacion.py", title="2. Segmentación K-Means"),
])

# Iconos de Lucide en el menú de páginas y en el botón de descarga (estos elementos no aceptan SVG directo)
st.html("<style>"
        + css_icono('[data-testid="stSidebarNavItems"] li:nth-child(1) a', "chart-column")
        + css_icono('[data-testid="stSidebarNavItems"] li:nth-child(2) a', "chart-scatter")
        + css_icono('[data-testid="stDownloadButton"] button', "download", "1rem")
        + "</style>")

encabezado("Segmentación de clientes de una tienda en línea con K-Means", "shopping-cart", 1, AZUL)
st.caption("Dataset *Online Retail*, UCI Machine Learning Repository (dic-2010 a dic-2011) · "
           "Usa el menú de la izquierda para cambiar de página.")

pagina.run()

st.caption("Proyecto integrador · Minería de datos · Fuente: Chen, D. (2015). Online Retail [Dataset]. "
           "UCI Machine Learning Repository. https://doi.org/10.24432/C5BW33 (CC BY 4.0)")
