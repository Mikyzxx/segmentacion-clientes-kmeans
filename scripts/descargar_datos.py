"""Descarga el dataset Online Retail (UCI) y lo guarda como CSV comprimido.

No modifica ningún valor: solo convierte el Excel original a `data/raw/online_retail.csv.gz`
para que el repositorio sea ligero y el notebook lo pueda leer con `pd.read_csv()`.

Uso (desde la raíz del repositorio):
    python scripts/descargar_datos.py
"""
import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

URL = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"
RAW = Path(__file__).resolve().parents[1] / "data" / "raw"


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    print(f"Descargando {URL} ...")
    with urllib.request.urlopen(URL) as resp:
        contenido = resp.read()
    with zipfile.ZipFile(io.BytesIO(contenido)) as z:
        z.extract("Online Retail.xlsx", RAW)
    print("Leyendo el Excel (puede tardar ~30 s)...")
    df = pd.read_excel(RAW / "Online Retail.xlsx", dtype={"InvoiceNo": str, "StockCode": str})
    destino = RAW / "online_retail.csv.gz"
    df.to_csv(destino, index=False, compression="gzip")
    print(f"Guardado {destino} ({len(df):,} filas x {df.shape[1]} columnas)")


if __name__ == "__main__":
    main()
