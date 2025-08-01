import streamlit as st
import base64
import time
from io import BytesIO
import requests

# ------------------
# Hugging Face API Config (Replace with your credentials)
# Recommended models:
# Text rewriting (tone adjustment): facebook/bart-large-cnn or meta-llama-3-8b-instruct
# TTS: espnet/kan-bayashi_ljspeech_vits or microsoft/speecht5_tts
# ------------------
HF_API_KEY = "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"
HF_TEXT_MODEL_URL = "https://api-inference.huggingface.co/models/meta-llama-3-8b-instruct"
HF_TTS_MODEL_URL = "https://api-inference.huggingface.co/models/microsoft/speecht5_tts"

# ------------------
# Hugging Face: Tone Rewriting
# ------------------
def rewrite_text_with_tone(text, tone):
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    payload = {"inputs": f"Rewrite the following text in a {tone} tone while preserving meaning: {text}"}
    try:
        response = requests.post(HF_TEXT_MODEL_URL, headers=headers, json=payload)
        result = response.json()
        if isinstance(result, list) and len(result) > 0:
            return result[0].get("generated_text", text)
        elif isinstance(result, dict) and "generated_text" in result:
            return result["generated_text"]
        return text
    except Exception as e:
        return f"[Error: {e}]\n{text}"

# ------------------
# Hugging Face: Voice Synthesis
# ------------------
def synthesize_speech(text, voice):
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    payload = {"inputs": text}
    try:
        response = requests.post(HF_TTS_MODEL_URL, headers=headers, json=payload)
        return BytesIO(response.content)
    except Exception as e:
        return BytesIO(b"")

# ------------------
# Streamlit UI Setup
# ------------------

st.set_page_config(page_title="EchoVerse", layout="wide")
page_bg_img = f"""
<style>
.stApp {{
    background-image: url('data:image/png;base64,{base64.b64encode(b'fake_bg_image').decode()}');
    background-size: cover;
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

st.title("🎧 EchoVerse – AI Audiobook Generator")
if "past_narrations" not in st.session_state:
    st.session_state.past_narrations = []

# ------------------
# Text Input + File Upload
# ------------------
st.subheader("📄 Upload or Enter Your Text")
uploaded_file = st.file_uploader("Upload a .txt file (max 200MB)", type=["txt"])
manual_text = st.text_area("Or paste your text here")
input_text = uploaded_file.read().decode("utf-8") if uploaded_file else manual_text

# ------------------
# Tone & Voice Selection
# ------------------
st.subheader("🎭 Select Tone & Voice")
tone = st.selectbox("Tone", ["Neutral", "Suspenseful", "Inspiring"])
voice = st.selectbox("Voice", ["Lisa", "Michael", "Allison", "Kate"])

# ------------------
# Generate Audiobook
# ------------------
if st.button("Generate Audiobook"):
    if input_text.strip() == "":
        st.warning("Please provide text input.")
    else:
        with st.spinner("Processing..."):
            rewritten_text = rewrite_text_with_tone(input_text, tone)
            audio_data = synthesize_speech(rewritten_text, voice)
            st.session_state.past_narrations.append({
                "original": input_text,
                "rewritten": rewritten_text,
                "voice": voice,
                "tone": tone,
                "audio": audio_data.getvalue()
            })
            col1, col2 = st.columns(2)
            col1.subheader("Original Text")
            col1.write(input_text)
            col2.subheader("Rewritten Text")
            col2.write(rewritten_text)
            st.audio(audio_data, format="audio/mp3")
            st.download_button("Download MP3", audio_data, file_name="audiobook.mp3")

# ------------------
# Past Narrations
# ------------------
with st.expander("📜 Past Narrations"):
    for idx, narration in enumerate(st.session_state.past_narrations):
        st.markdown(f"*Narration {idx+1} – {narration['tone']} Tone, {narration['voice']} Voice*")
        st.audio(BytesIO(narration["audio"]), format="audio/mp3")
        st.download_button(f"Download Narration {idx+1}", BytesIO(narration["audio"]), file_name=f"narration_{idx+1}.mp3")
