import streamlit as st
import pandas as pd
import plotly.express as px
import json
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from datetime import datetime
import logging

# ================= BASIC SETUP =================
st.set_page_config(page_title="Dashboard Interaktif Kabupaten Ponorogo", layout="wide")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(message)s")

st.title("Dashboard Interaktif Kabupaten Ponorogo — Auto Update")

# ================= PATH SETUP =================
data_folder = "C:/Users/otnie/Latihan/.vscode/Database"
os.makedirs(data_folder, exist_ok=True)

geojson_path = "C:/Users/otnie/Latihan/.vscode/35.02_Ponorogo/35.02_kecamatan.geojson"
geojson_key = "nm_kecamatan"

# ================= GEOJSON =================
geojson = None
if os.path.exists(geojson_path):
    try:
        with open(geojson_path, "r", encoding="utf-8") as f:
            geojson = json.load(f)
    except Exception as e:
        st.error(f"Gagal membaca GeoJSON: {e}")
        geojson = None
else:
    geojson = None

# ================= EXCEL → CSV =================
def convert_excel_to_csv(folder_path):
    excel_files = [f for f in os.listdir(folder_path) if f.endswith((".xlsx", ".xls"))]
    if excel_files:
        st.info("Ditemukan file Excel, mulai konversi ke CSV...")
        for excel_file in excel_files:
            excel_path = os.path.join(folder_path, excel_file)
            try:
                df = pd.read_excel(excel_path)
                output_name = os.path.splitext(excel_file)[0] + "_konversi.csv"
                output_path = os.path.join(folder_path, output_name)
                df.to_csv(output_path, sep=",", index=False, encoding="utf-8-sig")
                st.success(f"{excel_file} dikonversi menjadi {output_name}")
            except Exception as e:
                st.error(f"Gagal mengonversi {excel_file}: {e}")
    else:
        st.info("Tidak ada file Excel ditemukan, semua data sudah berformat CSV.")
    csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
    return csv_files

csv_files = convert_excel_to_csv(data_folder)

# ================= UTIL =================
def safe_read_csv(path):
    try:
        df = pd.read_csv(path)
        df.columns = [c.strip() for c in df.columns]
        return df
    except:
        return None
    
def generate_data_quality_report(df):
    report = {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "missing_per_col": df.isna().sum().to_dict(),
        "dtypes": df.dtypes.apply(lambda x: str(x)).to_dict(),
        "duplicate_rows": int(df.duplicated().sum())
    }
    report["numeric_summary"] = df.select_dtypes(include="number").describe().T.reset_index()
    return report

def impute_missing(df, strategy):
    df = df.copy()
    for c in df.select_dtypes(include="number").columns:
        if strategy == "mean":
            df[c].fillna(df[c].mean(), inplace=True)
        elif strategy == "median":
            df[c].fillna(df[c].median(), inplace=True)
        elif strategy == "zero":
            df[c].fillna(0, inplace=True)
    df.fillna("Unknown", inplace=True)
    return df

# ================= PROCESS DATA =================
def process_data(csv_name, n_clusters, contamination, impute_strategy=None, mask_identifiers=False):
    csv_path = os.path.join(data_folder, csv_name)
    df = safe_read_csv(csv_path)
    if df is None or "Nilai" not in df.columns or "Kecamatan" not in df.columns:
        st.error("Kolom 'Nilai' atau 'Kecamatan' tidak ditemukan.")
        return None

    df["Nilai"] = df["Nilai"].replace(["-", "", " "], pd.NA)
    df["Nilai"] = pd.to_numeric(df["Nilai"], errors="coerce")

    if df["Nilai"].isna().all():
        st.error("Semua nilai di kolom 'Nilai' kosong atau tidak valid.")
        return None

    if impute_strategy:
        df = impute_missing(df, impute_strategy)
    else:
        df = df.dropna(subset=["Nilai"])

    df = df.dropna(subset=["Nilai"])
    if df.shape[0] == 0:
        st.error("Tidak ada baris tersisa setelah pembersihan Nilai.")
        return None

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[["Nilai"]])

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df["Cluster"] = kmeans.fit_predict(X_scaled)
    cluster_order = df.groupby("Cluster")["Nilai"].mean().sort_values().index.tolist()
    label_map = {i: f"Klaster {rank+1}" for rank, i in enumerate(cluster_order)}
    df["Cluster_Label"] = df["Cluster"].map(label_map)

    iso = IsolationForest(contamination=contamination, random_state=42)
    df["Anomali"] = iso.fit_predict(df[["Nilai"]])
    df["Anomali"] = df["Anomali"].map({1: "Normal", -1: "Anomali"})

    return df.sort_values(by="Nilai", ascending=False).reset_index(drop=True)

# ================= SIDEBAR =================
st.sidebar.header("Pengaturan")

if not csv_files:
    st.sidebar.warning("Tidak ada file ditemukan di folder Database.")
    st.stop()

selected_csv = st.sidebar.selectbox("Pilih Dataset", csv_files)
n_clusters = st.sidebar.slider("Jumlah Klaster (KMeans)", 2, 8, 3)
contamination = st.sidebar.slider("Persentase Anomali (IsolationForest)", 0.01, 0.2, 0.05, 0.01)

impute_choice = st.sidebar.radio("Imputasi (Data Wrangling)", ["Tidak", "Mean", "Median", "Zero"])
if impute_choice == "Tidak":
    impute_strategy = None
elif impute_choice == "Mean":
    impute_strategy = "mean"
elif impute_choice == "Median":
    impute_strategy = "median"
else:
    impute_strategy = "zero"

map_style = st.sidebar.selectbox("Gaya Peta", ["carto-positron", "open-street-map", "carto-darkmatter"])

# ================= UPLOAD CSV / EXCEL =================
st.sidebar.markdown("---")
st.sidebar.header("Upload Dataset")

upload = st.sidebar.file_uploader(
    "Upload CSV / Excel",
    type=["csv", "xlsx", "xls"]
)

if upload:
    save_path = os.path.join(data_folder, upload.name)
    with open(save_path, "wb") as f:
        f.write(upload.getbuffer())

    if upload.name.endswith((".xlsx", ".xls")):
        df_excel = pd.read_excel(save_path)
        csv_name = os.path.splitext(upload.name)[0] + "_upload.csv"
        df_excel.to_csv(os.path.join(data_folder, csv_name),
                        index=False, encoding="utf-8-sig")

    st.sidebar.success("Dataset berhasil diupload")
    st.experimental_rerun()

# ================= RUN =================
df = process_data(selected_csv, n_clusters, contamination, impute_strategy)
if df is None:
    st.stop()

timestamp = f"Terakhir diperbarui: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
st.caption(timestamp)

# ================= KPI =================
total_kec = df["Kecamatan"].nunique()
mean_val = round(df["Nilai"].mean(), 2)
std_val = round(df["Nilai"].std(), 2)
anomaly_count = int((df["Anomali"] == "Anomali").sum())

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Kecamatan", total_kec)
col2.metric("Rata-rata Nilai", mean_val)
col3.metric("STD Nilai", std_val)
col4.metric("Jumlah Anomali", anomaly_count)

# ================= MAP =================
st.subheader("Peta Sebaran Nilai per Kecamatan")
if geojson:
    fig_map = px.choropleth_mapbox(
        df, geojson=geojson, locations="Kecamatan",
        featureidkey=f"properties.{geojson_key}",
        color="Cluster_Label",
        hover_data=["Kecamatan", "Nilai", "Cluster_Label", "Anomali"],
        mapbox_style=map_style, center={"lat": -7.97, "lon": 111.52}, zoom=9, height=520
    )
else:
    fig_map = px.bar(df, x="Kecamatan", y="Nilai", color="Cluster_Label", title="GeoJSON tidak ditemukan")

st.plotly_chart(fig_map, use_container_width=True)

# ----------------- Tabs -----------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Perbandingan (Bar)", "Distribusi (Histogram)", "Scatter", "Ringkasan Statistik", "Kualitas Data"])

with tab1:
    fig_bar = px.bar(df, x="Kecamatan", y="Nilai", color="Cluster_Label", title="Perbandingan Nilai per Kecamatan")
    st.plotly_chart(fig_bar, use_container_width=True)

with tab2:
    fig_hist = px.histogram(df, x="Nilai", nbins=20, title="Distribusi Nilai")
    st.plotly_chart(fig_hist, use_container_width=True)

with tab3:
    fig_scatter = px.scatter(df.reset_index(), x="index", y="Nilai", color="Cluster_Label", hover_data=["Kecamatan"], title="Scatter Nilai (index vs nilai)")
    st.plotly_chart(fig_scatter, use_container_width=True)

with tab4:
    summary = df.groupby("Cluster_Label")["Nilai"].agg(["mean", "max", "min"]).reset_index().round(2)
    st.dataframe(summary, use_container_width=True)

with tab5:
    dq = generate_data_quality_report(df)
    st.markdown("### Laporan Kualitas Data")
    dq_df = pd.DataFrame(list(dq["missing_per_col"].items()), columns=["Kolom", "Missing"])
    dq_df["Status"] = dq_df["Missing"].apply(lambda x: "Lengkap" if x == 0 else f"{x} data hilang")
    st.dataframe(dq_df, use_container_width=True)

    with st.expander("Ringkasan Tambahan"):
        st.write(f"**Jumlah Baris:** {dq['rows']}")
        st.write(f"**Jumlah Kolom:** {dq['columns']}")
        st.write(f"**Jumlah Duplikat:** {dq['duplicate_rows']}")
        dtype_df = pd.DataFrame(list(dq["dtypes"].items()), columns=["Kolom", "Tipe Data"])
        st.dataframe(dtype_df, use_container_width=True)

# ----------------- Insight -----------------
highest = df.iloc[0]
lowest = df.iloc[-1]
st.subheader("Insight Otomatis")
st.write(f"**Kecamatan nilai tertinggi:** {highest['Kecamatan']} ({highest['Nilai']}) — {highest['Cluster_Label']}")
st.write(f"**Kecamatan nilai terendah:** {lowest['Kecamatan']} ({lowest['Nilai']}) — {lowest['Cluster_Label']}")
st.write("**Ringkasan Statistik per Klaster:**")
st.dataframe(summary, use_container_width=True)

# ================= TABLE =================
st.subheader("Data Lengkap")
st.dataframe(df, use_container_width=True)