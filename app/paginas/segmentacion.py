"""Página 2 — Segmentación de clientes mediante K-Means (RFM)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from comun import (AZUL, COLOR_ICONO, COLORES, ETIQUETAS, GRIS, ICONOS, LOG_VARS, ORDEN, SIMBOLOS, TEXTO_2,  # noqa: E402
                   cargar_clientes, cargar_evaluacion, centroides, encabezado, estilo, icono, rango_log)

df = cargar_clientes()
evaluacion = cargar_evaluacion()

encabezado("Página 2 · Segmentación mediante K-Means", "chart-scatter", 2, AZUL)
st.caption(f"Análisis RFM (Recencia, Frecuencia, Monetario) de {len(df):,} clientes · K = 4 clústeres · "
           "variables transformadas con logaritmo y escaladas con StandardScaler.")

# ---------------------------------------------------------------------------
# Filtros (barra lateral)
with st.sidebar:
    st.header("Filtros · Segmentación")
    f_seg = st.multiselect("1. Segmento", ORDEN, default=ORDEN)
    f_reg = st.multiselect("2. Región", sorted(df["Region"].unique()),
                           default=sorted(df["Region"].unique()))
    paises_disp = sorted(df.loc[df["Region"].isin(f_reg), "Pais"].unique())
    f_pais = st.multiselect("3. País (opcional)", paises_disp, placeholder="Todos los países")
    f_mon = st.select_slider(
        "4. Gasto neto del cliente (£)",
        options=[0, 100, 250, 500, 1_000, 2_000, 5_000, 10_000, 50_000, 300_000],
        value=(0, 300_000), format_func=lambda v: f"£{v:,.0f}")
    f_rec = st.slider("5. Recencia (días desde la última compra)", 0, int(df["Recencia"].max()),
                      (0, int(df["Recencia"].max())))
    st.divider()
    st.caption("Los KPI, gráficas y tablas responden a los filtros. Los resultados del modelo, la descripción "
               "de los grupos y los hallazgos describen el modelo completo.")

mask = (df["Segmento"].isin(f_seg) & df["Region"].isin(f_reg)
        & df["Monetario"].between(*f_mon) & df["Recencia"].between(*f_rec))
if f_pais:
    mask &= df["Pais"].isin(f_pais)
dff = df[mask]

if dff.empty:
    st.warning("Ningún cliente cumple con los filtros seleccionados. Amplía los criterios en la barra lateral.")
    st.stop()

# ---------------------------------------------------------------------------
# KPI
k1, k2, k3, k4 = st.columns(4)
k1.metric("Clientes", f"{len(dff):,}", f"{len(dff) / len(df):.0%} del total", delta_color="off")
k2.metric("Ingresos netos", f"£{dff['Monetario'].sum() / 1e6:,.2f} M",
          f"{dff['Monetario'].sum() / df['Monetario'].sum():.0%} del total", delta_color="off")
k3.metric("Ticket promedio por compra", f"£{dff['Ventas'].sum() / dff['Frecuencia'].sum():,.0f}",
          help="Ventas totales / número total de facturas de los clientes filtrados")
k4.metric("Recencia mediana", f"{dff['Recencia'].median():.0f} días",
          help="Días transcurridos desde la última compra (cliente típico)")

st.divider()

# ---------------------------------------------------------------------------
# Distribución por clúster + dispersión
c1, c2 = st.columns([1, 1.6])
with c1:
    dist = (dff.groupby("Segmento", observed=False).size().reindex(ORDEN).fillna(0)
            .rename("Clientes").reset_index())
    dist["Pct"] = dist["Clientes"] / dist["Clientes"].sum() * 100
    fig = px.bar(dist, x="Clientes", y="Segmento", orientation="h", color="Segmento",
                 color_discrete_map=COLORES, category_orders={"Segmento": ORDEN},
                 text=dist.apply(lambda r: f"{r.Clientes:,.0f} ({r.Pct:.0f}%)", axis=1),
                 title="Distribución de clientes por clúster")
    fig.update_traces(textposition="outside", textfont_color=TEXTO_2, marker_line_width=0,
                      hovertemplate="<b>%{y}</b><br>%{x:,} clientes<extra></extra>")
    fig.update_layout(showlegend=False, bargap=0.35, yaxis_title=None)
    fig.update_xaxes(range=[0, dist["Clientes"].max() * 1.5])
    st.plotly_chart(estilo(fig, 400), width="stretch")

with c2:
    sx, sy = st.columns(2)
    var_x = sx.selectbox("Eje X", list(ETIQUETAS), index=0, format_func=ETIQUETAS.get)
    var_y = sy.selectbox("Eje Y", list(ETIQUETAS), index=2, format_func=ETIQUETAS.get)
    fig = px.scatter(dff, x=var_x, y=var_y, color="Segmento", symbol="Segmento",
                     color_discrete_map=COLORES, symbol_map=SIMBOLOS, category_orders={"Segmento": ORDEN},
                     opacity=0.75, hover_data={"CustomerID": True, "Pais": True, "Recencia": True,
                                               "Frecuencia": True, "Monetario": ":,.0f"},
                     labels=ETIQUETAS, title=f"Agrupamientos: {ETIQUETAS[var_x]} vs {ETIQUETAS[var_y]}")
    fig.update_traces(marker=dict(size=8, line=dict(width=1, color="white")))
    for var, eje in ((var_x, "x"), (var_y, "y")):
        if var in LOG_VARS:
            fig.update_layout(**{f"{eje}axis": dict(type="log", range=rango_log(dff[var]))})
    st.plotly_chart(estilo(fig, 400), width="stretch")
    st.caption("Las variables de dinero, frecuencia y productos se muestran en escala logarítmica, "
               "igual que en el modelo.")

# ---------------------------------------------------------------------------
# Comparación de características entre grupos
st.subheader("Comparación de características entre segmentos")
c3, c4 = st.columns(2)
with c3:
    var_box = st.selectbox("Variable a comparar", list(ETIQUETAS), index=2, format_func=ETIQUETAS.get)
    fig = px.box(dff, x="Segmento", y=var_box, color="Segmento", color_discrete_map=COLORES,
                 category_orders={"Segmento": ORDEN}, labels=ETIQUETAS, points=False,
                 title=f"Distribución de {ETIQUETAS[var_box].lower()} por segmento")
    if var_box in LOG_VARS:
        fig.update_yaxes(type="log", range=rango_log(dff[var_box]))
    fig.update_layout(showlegend=False, xaxis_title=None)
    st.plotly_chart(estilo(fig, 400), width="stretch")

with c4:
    part = dff.groupby("Segmento", observed=False).agg(Clientes=("CustomerID", "size"),
                                                       Ingresos=("Monetario", "sum")).reindex(ORDEN).fillna(0)
    part = (part / part.sum() * 100).reset_index().melt(id_vars="Segmento", var_name="Medida", value_name="Pct")
    part["Medida"] = part["Medida"].map({"Clientes": "% de clientes", "Ingresos": "% de ingresos"})
    st.markdown("<div style='height:4.6rem'></div>", unsafe_allow_html=True)
    fig = px.bar(part, x="Segmento", y="Pct", color="Medida", barmode="group",
                 color_discrete_map={"% de clientes": GRIS, "% de ingresos": AZUL},
                 category_orders={"Segmento": ORDEN}, text=part["Pct"].map("{:.0f}%".format),
                 title="Peso de cada segmento: clientes vs ingresos")
    fig.update_traces(textposition="outside", textfont_color=TEXTO_2, marker_line_width=0,
                      hovertemplate="<b>%{x}</b><br>%{fullData.name}: %{y:.1f}%<extra></extra>")
    fig.update_layout(yaxis_title="%", xaxis_title=None, bargap=0.3, bargroupgap=0.08)
    fig.update_yaxes(range=[0, part["Pct"].max() * 1.18])
    st.plotly_chart(estilo(fig, 400), width="stretch")

# Perfil relativo (mapa de calor)
perfil = dff.groupby("Segmento", observed=False)[list(ETIQUETAS)[:-1]].median().reindex(ORDEN).dropna()
indice = perfil / df[list(ETIQUETAS)[:-1]].median() * 100
fig = go.Figure(go.Heatmap(
    z=indice.values, x=[ETIQUETAS[c] for c in indice.columns], y=indice.index,
    text=indice.round(0).astype(int).astype(str).values, texttemplate="%{text}",
    colorscale=[[0, "#e34948"], [0.5, "#f1f0ec"], [1, AZUL]], zmid=100, zmin=0, zmax=300,
    colorbar=dict(title="Índice"),
    customdata=perfil.round(1).values,
    hovertemplate="<b>%{y}</b><br>%{x}<br>Mediana: %{customdata:,}<br>Índice: %{z:.0f}<extra></extra>"))
fig.update_layout(title="Perfil de cada segmento — mediana del grupo vs mediana general (100 = cliente típico)")
fig.update_yaxes(autorange="reversed")
st.plotly_chart(estilo(fig, 300), width="stretch")
st.caption("Azul = por encima del cliente típico · Rojo = por debajo. En Recencia un valor alto es negativo "
           "(más días sin comprar).")

# ---------------------------------------------------------------------------
# Tabla de estadísticas descriptivas
st.subheader("Estadísticas descriptivas por clúster")
vars_tabla = st.multiselect("Variables de la tabla", list(ETIQUETAS),
                            default=["Recencia", "Frecuencia", "Monetario", "TicketPromedio"],
                            format_func=ETIQUETAS.get)
if vars_tabla:
    est = dff.groupby("Segmento", observed=True)[vars_tabla].agg(["mean", "median", "std", "min", "max"])
    nombres_est = {"mean": "media", "median": "mediana", "std": "desv. est.", "min": "mín", "max": "máx"}
    est.columns = [f"{ETIQUETAS[v]} · {nombres_est[s]}" for v, s in est.columns]
    est.insert(0, "Clientes", dff.groupby("Segmento", observed=True).size())
    st.dataframe(est.style.format("{:,.1f}").format("{:,.0f}", subset=["Clientes"]), width="stretch")
st.download_button("Descargar clientes filtrados (CSV)", dff.to_csv(index=False).encode("utf-8"),
                   "clientes_filtrados.csv", "text/csv")

st.divider()

# ---------------------------------------------------------------------------
# Resultados del modelo
st.header("Resultados del modelo K-Means")
m1, m2 = st.columns(2)
fila4 = evaluacion.set_index("K").loc[4]
with m1:
    fig = px.line(evaluacion, x="K", y="Inercia", markers=True, title="Método del codo (inercia)")
    fig.add_vline(x=4, line_dash="dash", line_color=GRIS)
    fig.update_traces(line_color=GRIS, line_width=2, marker_size=8, marker_color=GRIS)
    fig.add_scatter(x=[4], y=[fila4["Inercia"]], mode="markers", marker=dict(size=13, color=AZUL),
                    showlegend=False, hovertemplate="K = 4<br>Inercia: %{y:,.0f}<extra></extra>")
    st.plotly_chart(estilo(fig, 320), width="stretch")
with m2:
    fig = px.line(evaluacion, x="K", y="Silueta", markers=True, title="Coeficiente de silueta (mayor = mejor)")
    fig.add_vline(x=4, line_dash="dash", line_color=GRIS)
    fig.update_traces(line_color=GRIS, line_width=2, marker_size=8, marker_color=GRIS)
    fig.add_scatter(x=[4], y=[fila4["Silueta"]], mode="markers", marker=dict(size=13, color=AZUL),
                    showlegend=False, hovertemplate="K = 4<br>Silueta: %{y:.3f}<extra></extra>")
    st.plotly_chart(estilo(fig, 320), width="stretch")
m3, m4 = st.columns([1.3, 1])
with m3:
    st.markdown("**Centroides en unidades originales**")
    cent = centroides().rename(columns={"Recencia (días)": "Recencia (d)", "Frecuencia (compras)": "Frecuencia"})
    st.dataframe(cent.style.format({"Recencia (d)": "{:,.0f}", "Frecuencia": "{:,.1f}",
                                    "Monetario (£)": "£{:,.0f}"}), width="stretch")
with m4:
    st.markdown("**Configuración elegida**")
    st.markdown(
        f"- **K elegido: 4** · silueta {fila4['Silueta']:.2f} · Davies-Bouldin {fila4['Davies-Bouldin']:.2f}\n"
        "- `KMeans(n_clusters=4, init='k-means++', n_init=10, random_state=42)`\n"
        "- Sin logaritmo la silueta sube a 0.61, pero un clúster queda con solo 6 clientes: por eso se "
        "eligió la versión con logaritmo.")
st.caption("Se eligió K = 4 por el codo de la inercia, por una silueta casi igual a K = 3 con mejor índice "
           "Davies-Bouldin, y porque cuatro grupos son útiles para el negocio.")

# ---------------------------------------------------------------------------
# Descripción de los grupos (modelo completo)
st.header("Descripción de los grupos identificados")
resumen = df.groupby("Segmento", observed=False).agg(
    Clientes=("CustomerID", "size"), Ingresos=("Monetario", "sum"), Recencia=("Recencia", "median"),
    Frecuencia=("Frecuencia", "median"), Monetario=("Monetario", "median"))
resumen["PctCli"] = resumen["Clientes"] / resumen["Clientes"].sum() * 100
resumen["PctIng"] = resumen["Ingresos"] / resumen["Ingresos"].sum() * 100

DESCRIPCIONES = {
    "Campeones": "Compraron hace muy poco, compran con mucha frecuencia y gastan varias veces más que el "
                 "cliente típico. Compran gran variedad de productos y casi todos llevan todo el año comprando.",
    "Leales en riesgo": "Clientes con historial y gasto por encima del promedio, pero su última compra fue "
                        "hace cerca de dos meses: se están enfriando y podrían pasar a inactivos.",
    "Nuevos / ocasionales": "Compraron recientemente pero solo una o dos veces y por montos bajos. La mayoría "
                            "son clientes nuevos que todavía no forman un hábito de compra.",
    "Inactivos": "Compraron una sola vez (o casi) hace aproximadamente seis meses y gastaron poco. Es el grupo "
                 "más grande, pero aporta una fracción pequeña de los ingresos.",
}
cols = st.columns(4)
for col, seg in zip(cols, ORDEN):
    r = resumen.loc[seg]
    with col.container(border=True):
        st.markdown(f"<div style='border-top:4px solid {COLORES[seg]};margin:-0.5rem -1rem 0.6rem'></div>",
                    unsafe_allow_html=True)
        encabezado(seg, ICONOS[seg], 3, COLOR_ICONO[seg])
        st.markdown(f"**{r.Clientes:,.0f} clientes ({r.PctCli:.0f}%) · {r.PctIng:.0f}% de los ingresos**")
        st.markdown(DESCRIPCIONES[seg])
        st.markdown(f"- Recencia mediana: **{r.Recencia:.0f} días**\n"
                    f"- Compras (mediana): **{r.Frecuencia:.0f}**\n"
                    f"- Gasto neto mediano: **£{r.Monetario:,.0f}**")

# ---------------------------------------------------------------------------
# Hallazgos y recomendaciones
st.header("Principales hallazgos y recomendaciones")
camp, ries, nuev, inac = (resumen.loc[s] for s in ORDEN)
top10 = df["Monetario"].nlargest(int(len(df) * .1)).sum() / df["Monetario"].sum()
una_vez = (df["Frecuencia"] == 1).mean()

h, r = st.columns(2)
with h:
    encabezado("Hallazgos", "search", 3, AZUL)
    st.markdown(f"""
1. **El ingreso está muy concentrado.** Los *Campeones* son el {camp.PctCli:.0f}% de los clientes y generan el
   **{camp.PctIng:.0f}% de los ingresos**; el 10% de clientes que más gasta concentra el {top10:.0%}.
2. **Hay ingresos importantes en riesgo.** Los *Leales en riesgo* ({ries.Clientes:,.0f} clientes) aportan el
   {ries.PctIng:.0f}% de los ingresos (£{ries.Ingresos / 1e6:,.2f} M), pero llevan una mediana de
   {ries.Recencia:.0f} días sin comprar.
3. **La retención es el principal problema.** El {inac.PctCli:.0f}% de los clientes está *Inactivo*
   (aporta solo el {inac.PctIng:.0f}% de los ingresos) y el {una_vez:.0%} de todos los clientes compró una
   sola vez en el año.
4. **Los Nuevos son la cantera de futuros clientes leales**: compraron hace {nuev.Recencia:.0f} días
   (mediana), el mejor momento para lograr la segunda compra.
5. **Perfil mayorista (B2B)**: compras de lunes a viernes en horario laboral y ~90% de los clientes en
   Reino Unido en todos los segmentos.
""")
with r:
    encabezado("Recomendaciones", "lightbulb", 3, AZUL)
    ic = {seg: icono(ICONOS[seg], "1.05em", COLOR_ICONO[seg], "0.15em") for seg in ORDEN}
    st.markdown(f"""
1. {ic["Campeones"]} **Campeones — Programa VIP:** ejecutivo de cuenta, descuentos por volumen, acceso anticipado a la
   temporada navideña y envío gratis. *Objetivo: retener y aumentar el ticket.*
2. {ic["Leales en riesgo"]} **Leales en riesgo — Reactivación temprana:** campaña personalizada antes de cumplir 60 días
   sin compra, con cupón de vigencia corta y recomendaciones según su historial. *Objetivo: evitar que pasen a
   inactivos.*
3. {ic["Nuevos / ocasionales"]} **Nuevos — Onboarding:** secuencia de bienvenida con incentivo para la **segunda compra** y
   catálogo de productos más vendidos. *Objetivo: convertirlos en recurrentes.*
4. {ic["Inactivos"]} **Inactivos — Recuperación de bajo costo:** correo automatizado antes de la temporada alta; si no
   responden, sacarlos de las campañas costosas.
5. **General:** asignar el presupuesto según el valor del segmento, crear alertas automáticas de recencia y
   reentrenar el modelo cada trimestre para medir cuántos clientes cambian de segmento.
""", unsafe_allow_html=True)


with st.expander("Limitaciones del modelo"):
    st.markdown("Datos de un solo año (2010-2011); se excluyó ~25% de transacciones sin `CustomerID`; K-Means "
                "supone grupos esféricos y es sensible a atípicos; la silueta (~0.34) indica clientes en las "
                "fronteras entre segmentos; solo se usaron variables RFM; el modelo describe, no predice el "
                "abandono.")
