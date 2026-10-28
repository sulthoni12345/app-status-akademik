import json

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Prediksi Status Akademik", page_icon="🎓")


# ---------- 1. Muat model & informasi model (sekali saja) ----------
@st.cache_resource
def load_model():
    return joblib.load("model_status_akademik.pkl")


@st.cache_data
def load_info():
    with open("model_info.json") as f:
        return json.load(f)


model = load_model()
info = load_info()
FITUR = info["fitur_numerik"] + info["fitur_kategori"]
rentang = info["rentang_fitur"]

# ---------- 2. Sidebar: informasi model ----------
with st.sidebar:
    st.header("Tentang Model")
    st.write(info["nama_model"])
    st.metric("Accuracy (data uji)", f"{info['metrik']['accuracy']:.1%}")
    st.metric("F1 macro (data uji)", f"{info['metrik']['f1_macro']:.3f}")
    st.caption(f"Dievaluasi pada {info['metrik']['n_test']} data uji · "
               f"scikit-learn {info['versi']['scikit-learn']}")

st.title("🎓 Prediksi Status Akademik Mahasiswa")
st.write("Masukkan data mahasiswa untuk memprediksi status: **Aman**, **Waspada**, atau **Berisiko**.")

tab1, tab2 = st.tabs(["Prediksi Satu Mahasiswa", "Prediksi dari File CSV"])

# ---------- 3. Prediksi satu data ----------
with tab1:
    with st.form("form_prediksi"):
        col1, col2 = st.columns(2)
        with col1:
            ipk = st.number_input("IPK", min_value=0.0, max_value=4.0, value=3.0, step=0.01)
            kehadiran = st.slider("Kehadiran (%)", 0, 100, 80)
            jam_belajar = st.slider("Jam belajar mandiri per minggu", 0, 40, 8)
        with col2:
            tugas = st.slider("Tugas terkumpul (%)", 0, 100, 80)
            uts = st.slider("Nilai UTS", 0, 100, 70)
            organisasi = st.radio("Aktif organisasi?", ["Ya", "Tidak"], horizontal=True)
        kirim = st.form_submit_button("Prediksi")

    if kirim:
        data = pd.DataFrame([{
            "ipk": ipk, "kehadiran": kehadiran, "jam_belajar": jam_belajar,
            "tugas_terkumpul": tugas, "nilai_uts": uts, "aktif_organisasi": organisasi,
        }])[FITUR]

        # peringatan jika input di luar rentang data latih
        luar = [c for c in rentang if not rentang[c][0] <= data[c].iloc[0] <= rentang[c][1]]
        if luar:
            st.warning(f"Nilai {', '.join(luar)} di luar rentang data latih; "
                       "hasil prediksi bisa kurang akurat.")

        hasil = model.predict(data)[0]
        proba = pd.Series(model.predict_proba(data)[0], index=model.classes_)

        warna = {"Aman": st.success, "Waspada": st.warning, "Berisiko": st.error}
        warna[hasil](f"Prediksi status: **{hasil}** (keyakinan {proba[hasil]:.0%})")
        st.bar_chart(proba.rename("probabilitas"))

# ---------- 4. Prediksi banyak data dari CSV ----------
with tab2:
    st.write("Unggah file CSV dengan kolom: " + ", ".join(f"`{c}`" for c in FITUR))
    file = st.file_uploader("Pilih file CSV", type="csv")
    if file is not None:
        df = pd.read_csv(file)
        kurang = [c for c in FITUR if c not in df.columns]
        if kurang:
            st.error(f"Kolom berikut tidak ditemukan: {', '.join(kurang)}")
        else:
            df["prediksi_status"] = model.predict(df[FITUR])
            st.dataframe(df)
            st.write(df["prediksi_status"].value_counts())
            st.download_button("Unduh hasil prediksi", df.to_csv(index=False),
                               file_name="hasil_prediksi.csv", mime="text/csv")