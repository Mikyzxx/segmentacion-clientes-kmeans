"""Página 1 — Análisis general del dataset (líneas de venta limpias)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from comun import (AZUL, DIAS, GRIS, MESES, TEXTO_2, cargar_bitacora, cargar_ventas, encabezado,  # noqa: E402
                   estilo)

ventas = cargar_ventas()

encabezado("Página 1 · Análisis general del dataset", "chart-column", 2, AZUL)
st.caption(f"Indicadores y visualizaciones de {len(ventas):,} líneas de venta limpias: sin duplicados, sin "
           "registros sin cliente, sin códigos que no son productos y sin los pedidos que después se cancelaron "
           "(por ejemplo, un pedido de 80,995 unidades cancelado el mismo día).")

# ---------------------------------------------------------------------------
# Filtros
meses = sorted(ventas["Mes"].unique())
etiqueta_mes = lambda m: f"{MESES[pd.Timestamp(m).month - 1]} {pd.Timestamp(m).year}"  # noqa: E731
with st.sidebar:
    st.header("Filtros · Análisis general")
    f_periodo = st.select_slider("1. Periodo", options=meses, value=(meses[0], meses[-1]),
                                 format_func=etiqueta_mes)
    regiones = sorted(ventas["Region"].unique())
    f_reg = st.multiselect("2. Región", regiones, default=regiones)
    paises_disp = sorted(ventas.loc[ventas["Region"].isin(f_reg), "Country"].unique())
    f_pais = st.multiselect("3. País (opcional)", paises_disp, placeholder="Todos los países")
    st.divider()
    st.caption("Todos los indicadores, gráficas y tablas de esta página responden a los filtros.")

mask = ventas["Mes"].between(*f_periodo) & ventas["Region"].isin(f_reg)
if f_pais:
    mask &= ventas["Country"].isin(f_pais)
v = ventas[mask]
if v.empty:
    st.warning("No hay ventas con los filtros seleccionados. Amplía los criterios en la barra lateral.")
    st.stop()

facturas = v.groupby("InvoiceNo").agg(Total=("Total", "sum"), Fecha=("InvoiceDate", "first"))

# ---------------------------------------------------------------------------
# KPI
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Ingresos", f"£{v['Total'].sum() / 1e6:,.2f} M",
          f"{v['Total'].sum() / ventas['Total'].sum():.0%} del total", delta_color="off")
k2.metric("Facturas", f"{len(facturas):,}")
k3.metric("Clientes", f"{v['CustomerID'].nunique():,}")
k4.metric("Valor promedio por factura", f"£{facturas['Total'].mean():,.0f}",
          help="Ingresos / número de facturas")
k5.metric("Productos distintos", f"{v['StockCode'].nunique():,}")

st.divider()

# ---------------------------------------------------------------------------
# Evolución mensual (dos gráficas, cada una con su propio eje)
c1, c2 = st.columns(2)
mensual = v.groupby("Mes").agg(Ingresos=("Total", "sum"), Clientes=("CustomerID", "nunique")).reset_index()
mensual["Etiqueta"] = mensual["Mes"].map(etiqueta_mes)
mensual["Parcial"] = mensual["Mes"] == pd.Timestamp("2011-12-01")
with c1:
    fig = px.bar(mensual, x="Etiqueta", y="Ingresos", title="Ingresos por mes (£)")
    fig.update_traces(marker_color=[GRIS if p else AZUL for p in mensual["Parcial"]], marker_line_width=0,
                      hovertemplate="<b>%{x}</b><br>£%{y:,.0f}<extra></extra>")
    fig.update_layout(xaxis_title=None, yaxis_title="£", bargap=0.25)
    st.plotly_chart(estilo(fig, 360), width="stretch")
with c2:
    fig = px.bar(mensual, x="Etiqueta", y="Clientes", title="Clientes que compraron cada mes")
    fig.update_traces(marker_color=[GRIS if p else AZUL for p in mensual["Parcial"]], marker_line_width=0,
                      hovertemplate="<b>%{x}</b><br>%{y:,} clientes<extra></extra>")
    fig.update_layout(xaxis_title=None, yaxis_title="Clientes", bargap=0.25)
    st.plotly_chart(estilo(fig, 360), width="stretch")
st.caption("Las ventas crecen de septiembre a noviembre por la temporada navideña. Diciembre de 2011 (en gris) "
           "solo tiene 9 días de datos.")

# ---------------------------------------------------------------------------
# Países y productos
c3, c4 = st.columns(2)
with c3:
    excluir_uk = st.checkbox("Excluir Reino Unido del ranking de países", value=False)
    paises = v if not excluir_uk else v[v["Country"] != "United Kingdom"]
    top_p = paises.groupby("Country")["Total"].sum().nlargest(10).sort_values().reset_index()
    if top_p.empty:
        st.info("No hay países que mostrar con estos filtros.")
    else:
        fig = px.bar(top_p, x="Total", y="Country", orientation="h", title="Top 10 países por ingresos (£)",
                     text=top_p["Total"].map(lambda x: f"£{x / 1e3:,.0f}k"))
        fig.update_traces(marker_color=AZUL, marker_line_width=0, textposition="outside",
                          textfont_color=TEXTO_2, hovertemplate="<b>%{y}</b><br>£%{x:,.0f}<extra></extra>")
        fig.update_layout(xaxis_title="£", yaxis_title=None, bargap=0.3)
        fig.update_xaxes(range=[0, top_p["Total"].max() * 1.2])
        st.plotly_chart(estilo(fig, 400), width="stretch")
with c4:
    st.markdown("<div style='height:2.4rem'></div>", unsafe_allow_html=True)
    top_prod = (v.groupby("Description")["Total"].sum().nlargest(10).sort_values().reset_index())
    top_prod["Nombre"] = top_prod["Description"].str.title().str.slice(0, 34)
    fig = px.bar(top_prod, x="Total", y="Nombre", orientation="h", title="Top 10 productos por ingresos (£)",
                 text=top_prod["Total"].map(lambda x: f"£{x / 1e3:,.0f}k"), hover_data={"Description": True})
    fig.update_traces(marker_color=AZUL, marker_line_width=0, textposition="outside", textfont_color=TEXTO_2)
    fig.update_layout(xaxis_title="£", yaxis_title=None, bargap=0.3)
    fig.update_xaxes(range=[0, top_prod["Total"].max() * 1.2])
    st.plotly_chart(estilo(fig, 400), width="stretch")

# ---------------------------------------------------------------------------
# Patrón semanal y distribución del valor de factura
c5, c6 = st.columns(2)
with c5:
    tabla = (v.pivot_table(index="DiaSemana", columns="Hora", values="InvoiceNo", aggfunc="nunique", fill_value=0))
    fig = go.Figure(go.Heatmap(
        z=tabla.values, x=[f"{h}:00" for h in tabla.columns], y=[DIAS[d] for d in tabla.index],
        colorscale=[[0, "#f1f0ec"], [1, AZUL]], colorbar=dict(title="Facturas"),
        hovertemplate="<b>%{y} %{x}</b><br>%{z:,} facturas<extra></extra>"))
    fig.update_layout(title="Facturas por día de la semana y hora")
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(estilo(fig, 380), width="stretch")
    st.caption("No hay ventas en sábado y la actividad se concentra entre 10:00 y 15:00: un patrón típico de "
               "clientes empresariales.")
with c6:
    fig = px.histogram(np.log10(facturas["Total"].clip(lower=1)), nbins=50,
                       title="Distribución del valor de las facturas")
    fig.update_traces(marker_color=AZUL, marker_line_width=0,
                      hovertemplate="log10(£) = %{x:.2f}<br>%{y:,} facturas<extra></extra>")
    ticks = [1, 10, 100, 1_000, 10_000, 100_000]
    fig.update_layout(showlegend=False, xaxis_title="Valor de la factura (£, escala logarítmica)",
                      yaxis_title="Facturas", bargap=0.05)
    fig.update_xaxes(tickvals=np.log10(ticks), ticktext=[f"£{t:,}" for t in ticks])
    st.plotly_chart(estilo(fig, 380), width="stretch")
    st.caption(f"La mitad de las facturas vale menos de £{facturas['Total'].median():,.0f}, pero hay pedidos "
               "mayoristas de miles de libras: la distribución está muy sesgada.")

# ---------------------------------------------------------------------------
# Estadísticas descriptivas
st.subheader("Estadísticas descriptivas")
desc = pd.concat({
    "Unidades por línea": v["Quantity"].describe(),
    "Precio unitario (£)": v["UnitPrice"].describe(),
    "Importe por línea (£)": v["Total"].describe(),
    "Valor por factura (£)": facturas["Total"].describe(),
}, axis=1).T
desc.columns = ["Registros", "Media", "Desv. est.", "Mín", "25%", "Mediana", "75%", "Máx"]
st.dataframe(desc.style.format("{:,.2f}").format("{:,.0f}", subset=["Registros"]), width="stretch")
st.caption("Estadísticas de las líneas de venta y de las facturas que cumplen los filtros.")

st.subheader("Limpieza de datos realizada en Python")
e1, e2 = st.columns([1.6, 1])
with e1:
    bit = cargar_bitacora()
    st.dataframe(bit.style.format("{:,.0f}", subset=["Antes", "Después", "Eliminados"]),
                 hide_index=True, width="stretch")
with e2:
    cortos = ["Sin duplicados", "Con cliente", "Solo productos", "Precio > 0", "Sin devoluciones"]
    emb = pd.DataFrame({"Paso": ["Original"] + cortos,
                        "Registros": [bit["Antes"].iloc[0]] + bit["Después"].tolist()}).iloc[::-1]
    fig = px.bar(emb, x="Registros", y="Paso", orientation="h", title="Registros que quedan tras cada paso",
                 text=emb["Registros"].map(lambda x: f"{x:,.0f}"))
    fig.update_traces(marker_color=[GRIS if p == "Original" else AZUL for p in emb["Paso"]], marker_line_width=0,
                      textposition="inside", insidetextanchor="end", textfont_color="white",
                      hovertemplate="<b>%{y}</b><br>%{x:,.0f} registros<extra></extra>")
    fig.update_layout(xaxis_title=None, yaxis_title=None, bargap=0.3)
    st.plotly_chart(estilo(fig, 290), width="stretch")
st.caption("De 541,909 registros originales a 391,150 líneas de venta válidas (dataset completo, sin filtros). "
           "El paso que más registros elimina es quitar las ventas sin cliente identificado.")
