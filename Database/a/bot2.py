import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# -------------------------------
# 1️⃣ Setup Halaman Streamlit
# -------------------------------
st.set_page_config(page_title="Chatbot Indonesia Gratis 🤖", page_icon="🤖", layout="centered")
st.title("🤖 Chatbot Bahasa Indonesia Gratis 🇮🇩")
st.write("Ngobrol santai dengan AI lokal yang bisa berbahasa Indonesia tanpa perlu API 🔥")

# -------------------------------
# 2️⃣ Muat Model & Tokenizer
# -------------------------------
@st.cache_resource
def load_model():
    model_name = "flax-community/gpt2-medium-indonesian"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# -------------------------------
# 3️⃣ Inisialisasi Riwayat Chat
# -------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# -------------------------------
# 4️⃣ Input dari Pengguna
# -------------------------------
user_input = st.chat_input("Ketik pesanmu di sini...")

def generate_response(prompt, history_text):
    # Gabungkan riwayat + input terbaru
    input_text = history_text + "\nUser: " + prompt + "\nBot:"
    inputs = tokenizer.encode(input_text, return_tensors="pt")

    outputs = model.generate(
        inputs,
        max_length=250,
        pad_token_id=tokenizer.eos_token_id,
        temperature=0.8,
        top_p=0.9,
        do_sample=True,
    )

    reply = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Ambil hanya jawaban setelah "Bot:"
    reply = reply.split("Bot:")[-1].strip()
    return reply

# -------------------------------
# 5️⃣ Proses Saat User Kirim Pesan
# -------------------------------
if user_input:
    history_text = "\n".join(
        [f"User: {u}\nBot: {b}" for u, b in st.session_state.history]
    )
    bot_reply = generate_response(user_input, history_text)

    st.session_state.history.append((user_input, bot_reply))

# -------------------------------
# 6️⃣ Tampilkan Percakapan
# -------------------------------
for user, bot in st.session_state.history:
    st.chat_message("user").markdown(user)
    st.chat_message("assistant").markdown(bot)

# -------------------------------
# 7️⃣ Tombol Hapus Riwayat
# -------------------------------
if st.button("🔄 Hapus Riwayat"):
    st.session_state.history = []
    st.rerun()
