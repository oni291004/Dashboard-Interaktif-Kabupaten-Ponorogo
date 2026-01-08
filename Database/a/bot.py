import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import os

# -------------------------------
# 1️⃣ Muat API Key dari .env
# -------------------------------
load_dotenv()  # baca file .env
api_key = os.getenv("OPENAI_API_KEY")  # ambil dari .env, bukan langsung string!

if not api_key:
    st.error("❌ OPENAI_API_KEY belum diatur di file .env")
    st.stop()

client = OpenAI(api_key=api_key)

# -------------------------------
# 2️⃣ Setup halaman Streamlit
# -------------------------------
st.set_page_config(page_title="Chatbot AI Bahasa Indonesia 🤖", page_icon="🤖", layout="centered")
st.title("🤖 Chatbot AI Bahasa Indonesia 🇮🇩")
st.write("Ngobrol langsung dengan AI yang paham Bahasa Indonesia 🇮🇩")

# -------------------------------
# 3️⃣ Inisialisasi sesi obrolan
# -------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "Kamu adalah asisten AI yang menjawab dalam Bahasa Indonesia dengan sopan, natural, dan jelas."}
    ]

# Tampilkan riwayat obrolan
for msg in st.session_state.messages:
    if msg["role"] != "system":
        with st.chat_message("user" if msg["role"] == "user" else "assistant"):
            st.markdown(msg["content"])

# -------------------------------
# 4️⃣ Input pengguna
# -------------------------------
prompt = st.chat_input("Tulis pertanyaanmu di sini...")

if prompt:
    # Simpan pesan user
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").markdown(prompt)

    # -------------------------------
    # 5️⃣ Kirim ke OpenAI API
    # -------------------------------
    with st.chat_message("assistant"):
        with st.spinner("AI sedang mengetik..."):
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",  # model cepat & murah
                    messages=st.session_state.messages,
                    temperature=0.8  # lebih natural
                )
                reply = response.choices[0].message.content
                st.markdown(reply)
                st.session_state.messages.append({"role": "assistant", "content": reply})
            except Exception as e:
                st.error(f"❌ Terjadi kesalahan: {e}")

# -------------------------------
# 6️⃣ Tombol hapus riwayat chat
# -------------------------------
if st.button("🔄 Hapus Riwayat Chat"):
    st.session_state.messages = [
        {"role": "system", "content": "Kamu adalah asisten AI yang menjawab dalam Bahasa Indonesia dengan sopan, natural, dan jelas."}
    ]
    st.rerun()
