import streamlit as st
import torch
import soundfile as sf
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan, MarianMTModel, MarianTokenizer
import io
import time
import random
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="EchoVerse Audiobook Creator",
    page_icon="📚",
    layout="wide"
)

# --- Theming ---
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

# --- Load TTS Models ---
@st.cache_resource
def load_tts_models_and_embeddings():
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

    voice_mapping = {
        "Lisa": torch.load("lisa_emb.pt"),
        "Michael": torch.load("michael_emb.pt"),
        "Allison": torch.load("allison_emb.pt"),
        "Kate": torch.load("kate_emb.pt"),
    }
    return processor, model, vocoder, voice_mapping

# --- Load Translation Models ---
@st.cache_resource
def load_translation_models():
    langs = {
        "Spanish": "Helsinki-NLP/opus-mt-en-es",
        "French": "Helsinki-NLP/opus-mt-en-fr",
        "German": "Helsinki-NLP/opus-mt-en-de",
        "Hindi": "Helsinki-NLP/opus-mt-en-hi",
        "Telugu": "Helsinki-NLP/opus-mt-en-te"
    }
    models = {}
    tokenizers = {}
    for lang, model_name in langs.items():
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        models[lang] = model
        tokenizers[lang] = tokenizer
    return models, tokenizers

# --- Tone Rewriting ---
def rewrite_text_with_tone(text, tone):
    if tone == "Inspiring":
        return f"Imagine a world where {text.lower()} This truly shows what we can achieve together."
    elif tone == "Suspenseful":
        return f"In the shadows, a secret was kept... {text} But what they didn't know was far more terrifying."
    return text

# --- Translate ---
def translate_text(text, target_lang, translation_models, translation_tokenizers):
    if target_lang == "English":
        return text
    tokenizer = translation_tokenizers[target_lang]
    model = translation_models[target_lang]
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    translated_tokens = model.generate(**inputs)
    return tokenizer.decode(translated_tokens[0], skip_special_tokens=True)

# --- Session State ---
if 'narrations' not in st.session_state:
    st.session_state.narrations = []

# --- Main UI ---
st.title("📚 EchoVerse Audiobook Creator")

with st.container():
    st.header("1. Provide Your Text")
    uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])
    text_input = st.text_area("Or paste your text here", height=150, placeholder="Enter your content...")
    
    original_text = ""
    if uploaded_file:
        try:
            original_text = uploaded_file.getvalue().decode("utf-8")
        except Exception as e:
            st.error(f"Error reading file: {e}")
    elif text_input:
        original_text = text_input

# --- Load Models ---
with st.spinner("Loading AI engines..."):
    processor, model, vocoder, voice_mapping = load_tts_models_and_embeddings()
    translation_models, translation_tokenizers = load_translation_models()

with st.container():
    st.header("2. Customize Your Audiobook")
    col1, col2, col3 = st.columns(3)
    with col1:
        selected_tone = st.selectbox("Narrative Tone", ["Neutral", "Inspiring", "Suspenseful"])
    with col2:
        selected_voice = st.selectbox("Voice Style", list(voice_mapping.keys()))
    with col3:
        selected_lang = st.selectbox("Language", ["English", "Spanish", "French", "German", "Hindi", "Telugu"])

# --- Generate Audiobook ---
if st.button("Generate Audiobook", type="primary"):
    if not original_text.strip():
        st.warning("Please provide some text.")
    else:
        with st.spinner(f"Applying {selected_tone} tone..."):
            time.sleep(random.uniform(1.5, 2.5))
            rewritten_text = rewrite_text_with_tone(original_text, selected_tone)

        with st.spinner(f"Translating to {selected_lang}..."):
            translated_text = translate_text(rewritten_text, selected_lang, translation_models, translation_tokenizers)

        st.header("Original / Rewritten / Translated")
        st.markdown(f"**Original:** {original_text}")
        st.markdown(f"**Rewritten ({selected_tone}):** {rewritten_text}")
        st.markdown(f"**Translated ({selected_lang}):** {translated_text}")

        with st.spinner(f"Synthesizing {selected_lang} audio..."):
            try:
                inputs = processor(text=translated_text, return_tensors="pt")
                speaker_embedding = voice_mapping[selected_voice]
                speech = model.generate_speech(
                    inputs["input_ids"],
                    speaker_embeddings=speaker_embedding,
                    vocoder=vocoder
                )
                audio_np = speech.numpy()

                st.audio(audio_np, sample_rate=16000)
                buffer = io.BytesIO()
                sf.write(buffer, audio_np, 16000, format='WAV')
                audio_bytes = buffer.getvalue()

                st.download_button(
                    label="Download MP3",
                    data=audio_bytes,
                    file_name=f"EchoVerse_{selected_tone}_{selected_voice}_{selected_lang}.mp3",
                    mime="audio/mpeg"
                )

                st.session_state.narrations.append({
                    "rewritten_text": rewritten_text,
                    "translated_text": translated_text,
                    "tone": selected_tone,
                    "voice": selected_voice,
                    "lang": selected_lang,
                    "audio_bytes": audio_bytes
                })
            except Exception as e:
                st.error(f"TTS Error: {e}")

# --- Past Narrations ---
if st.session_state.narrations:
    st.divider()
    with st.expander("📖 Past Narrations"):
        for i, narration in enumerate(reversed(st.session_state.narrations)):
            st.markdown(f"**Narration {len(st.session_state.narrations)-i}** | Tone: `{narration['tone']}` | Voice: `{narration['voice']}` | Lang: `{narration['lang']}`")
            st.audio(narration['audio_bytes'], sample_rate=16000)
            st.download_button(
                label="Re-download",
                data=narration['audio_bytes'],
                file_name=f"EchoVerse_{narration['tone']}_{narration['voice']}_{narration['lang']}.mp3",
                mime="audio/mpeg",
                key=f"download_{i}"
            )
