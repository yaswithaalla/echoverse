import streamlit as st
import torch
import soundfile as sf
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import io
import time
import random

# --- Page Configuration ---
st.set_page_config(
    page_title="EchoVerse Audiobook Creator",
    page_icon="📚",
    layout="wide"
)

# --- Visuals & Theming (Component 6) ---
st.markdown("""
<style>
    .stApp { background-color: #e0e7ff; }
    [data-testid="stVerticalBlock"] .st-emotion-cache-16txtl3 {
        background-color: rgba(255, 255, 255, 0.8);
        border-radius: 0.5rem;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    h1 { color: #312e81; }
</style>
""", unsafe_allow_html=True)

# --- Load Models & Embeddings ---
@st.cache_resource
def load_tts_models_and_embeddings():
    """Load SpeechT5 models and local speaker embeddings."""
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

    # Load local embeddings saved from save_embeddings_local.py
    voice_mapping = {
        "Lisa": torch.load("lisa_emb.pt"),
        "Michael": torch.load("michael_emb.pt"),
        "Allison": torch.load("allison_emb.pt"),
        "Kate": torch.load("kate_emb.pt"),
    }
    return processor, model, vocoder, voice_mapping

# --- Tone Rewriting Simulation ---
def rewrite_text_with_tone(text, tone):
    """Simulates tone rewriting for Neutral, Inspiring, Suspenseful."""
    if tone == "Inspiring":
        return f"Imagine a world where {text.lower()} This truly shows what we can achieve together."
    elif tone == "Suspenseful":
        return f"In the shadows, a secret was kept... {text} But what they didn't know was far more terrifying."
    return text  # Neutral tone

# --- Session State ---
if 'narrations' not in st.session_state:
    st.session_state.narrations = []

# --- Main UI ---
st.title("📚 EchoVerse Audiobook Creator")

with st.container():
    st.header("1. Provide Your Text")
    uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])
    text_input = st.text_area("Or paste your text here", height=150, placeholder="Enter any content for transformation...")
    
    original_text = ""
    if uploaded_file:
        try:
            original_text = uploaded_file.getvalue().decode("utf-8")
        except Exception as e:
            st.error(f"Error reading file: {e}")
    elif text_input:
        original_text = text_input

# --- Load Models ---
with st.spinner("Warming up the AI engines..."):
    processor, model, vocoder, voice_mapping = load_tts_models_and_embeddings()

with st.container():
    st.header("2. Customize Your Audiobook")
    col1, col2 = st.columns(2)
    with col1:
        selected_tone = st.selectbox("Select Narrative Tone", ["Neutral", "Inspiring", "Suspenseful"])
    with col2:
        selected_voice = st.selectbox("Select Voice Style", list(voice_mapping.keys()))

# --- Generate Audiobook ---
if st.button("Generate Audiobook", type="primary"):
    if not original_text.strip():
        st.warning("Please provide some text first.")
    else:
        with st.spinner(f"Rewriting text in {selected_tone} tone..."):
            time.sleep(random.uniform(1.5, 2.5))
            rewritten_text = rewrite_text_with_tone(original_text, selected_tone)

        st.header("Original vs. Rewritten Text")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original")
            st.markdown(f"> {original_text}")
        with col2:
            st.subheader("Rewritten")
            st.markdown(f"> {rewritten_text}")
        
        st.divider()

        with st.spinner(f"Synthesizing audio in {selected_voice}'s voice..."):
            try:
                inputs = processor(text=rewritten_text, return_tensors="pt")
                speaker_embedding = voice_mapping[selected_voice]

                speech = model.generate_speech(
                    inputs["input_ids"],
                    speaker_embeddings=speaker_embedding,
                    vocoder=vocoder
                )
                audio_np = speech.numpy()

                st.header("🎧 Your Generated Audiobook")
                st.audio(audio_np, sample_rate=16000)

                buffer = io.BytesIO()
                sf.write(buffer, audio_np, 16000, format='WAV')
                audio_bytes = buffer.getvalue()

                st.download_button(
                    label="Download MP3",
                    data=audio_bytes,
                    file_name=f"EchoVerse_{selected_tone}_{selected_voice}.mp3",
                    mime="audio/mpeg"
                )

                st.session_state.narrations.append({
                    "rewritten_text": rewritten_text,
                    "tone": selected_tone,
                    "voice": selected_voice,
                    "audio_bytes": audio_bytes
                })
            except Exception as e:
                st.error(f"TTS Error: {e}")

# --- Past Narrations ---
if st.session_state.narrations:
    st.divider()
    with st.expander("📖 Past Narrations"):
        for i, narration in enumerate(reversed(st.session_state.narrations)):
            st.markdown(f"**Narration {len(st.session_state.narrations)-i}** | Tone: `{narration['tone']}` | Voice: `{narration['voice']}`")
            st.audio(narration['audio_bytes'], sample_rate=16000)
            st.markdown(f"> _{narration['rewritten_text'][:150]}..._")
            st.download_button(
                label="Re-download",
                data=narration['audio_bytes'],
                file_name=f"EchoVerse_{narration['tone']}_{narration['voice']}.mp3",
                mime="audio/mpeg",
                key=f"download_{i}"
            )
            st.markdown("---")

