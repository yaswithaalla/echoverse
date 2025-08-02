import streamlit as st
import torch
import soundfile as sf
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan, pipeline
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
    .summary-box {
        background-color: #f0f9ff;
        border-left: 4px solid #0ea5e9;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.5rem;
    }
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

@st.cache_resource
def load_summarizer():
    """Load summarization model."""
    try:
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        return summarizer
    except Exception as e:
        st.error(f"Error loading summarizer: {e}")
        return None

# --- Text Processing Functions ---
def summarize_text(text, summarizer, max_length=150, min_length=50):
    """Summarize text using BART model."""
    if not summarizer:
        return text
    
    try:
        # Split long text into chunks if needed
        max_chunk_length = 1024
        if len(text) > max_chunk_length:
            chunks = [text[i:i+max_chunk_length] for i in range(0, len(text), max_chunk_length)]
            summaries = []
            for chunk in chunks:
                if len(chunk.strip()) > 50:  # Only summarize if chunk is substantial
                    summary = summarizer(chunk, max_length=max_length//len(chunks), min_length=min_length//len(chunks), do_sample=False)
                    summaries.append(summary[0]['summary_text'])
                else:
                    summaries.append(chunk)
            return " ".join(summaries)
        else:
            summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
            return summary[0]['summary_text']
    except Exception as e:
        st.error(f"Summarization error: {e}")
        return text

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
    summarizer = load_summarizer()

with st.container():
    st.header("2. Text Processing Options")
    
    # Text processing options
    col1, col2 = st.columns(2)
    with col1:
        use_summarizer = st.checkbox("📝 Summarize text first", help="Create a concise summary before generating audio")
        if use_summarizer:
            summary_length = st.slider("Summary length", min_value=50, max_value=300, value=150, step=25)
    
    with col2:
        selected_tone = st.selectbox("Select Narrative Tone", ["Neutral", "Inspiring", "Suspenseful"])

    st.header("3. Voice Customization")
    selected_voice = st.selectbox("Select Voice Style", list(voice_mapping.keys()))

# --- Text Preview Section ---
if original_text.strip():
    st.header("📖 Text Preview")
    
    # Show summarized version if option is selected
    if use_summarizer and summarizer:
        with st.spinner("Generating summary..."):
            time.sleep(random.uniform(0.5, 1.0))  # Simulate processing time
            summarized_text = summarize_text(original_text, summarizer, max_length=summary_length)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📄 Original Text")
            st.markdown(f'<div class="summary-box">{original_text[:300]}{"..." if len(original_text) > 300 else ""}</div>', unsafe_allow_html=True)
            st.caption(f"Length: {len(original_text.split())} words")
        
        with col2:
            st.subheader("📋 Summary")
            st.markdown(f'<div class="summary-box">{summarized_text}</div>', unsafe_allow_html=True)
            st.caption(f"Length: {len(summarized_text.split())} words")
        
        # Use summarized text for further processing
        processed_text = summarized_text
    else:
        processed_text = original_text
        st.markdown(f'<div class="summary-box">{original_text[:500]}{"..." if len(original_text) > 500 else ""}</div>', unsafe_allow_html=True)

# --- Generate Audiobook ---
if st.button("🎵 Generate Audiobook", type="primary"):
    if not original_text.strip():
        st.warning("Please provide some text first.")
    else:
        # Determine which text to use
        if use_summarizer and summarizer:
            if 'summarized_text' not in locals():
                with st.spinner("Generating summary..."):
                    summarized_text = summarize_text(original_text, summarizer, max_length=summary_length)
            final_text = summarized_text
        else:
            final_text = original_text

        with st.spinner(f"Rewriting text in {selected_tone} tone..."):
            time.sleep(random.uniform(1.5, 2.5))
            rewritten_text = rewrite_text_with_tone(final_text, selected_tone)

        st.header("📝 Text Transformation Pipeline")
        
        # Show the transformation pipeline
        if use_summarizer and summarizer:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.subheader("Original")
                st.markdown(f"> {original_text[:150]}...")
            with col2:
                st.subheader("Summarized")
                st.markdown(f"> {final_text}")
            with col3:
                st.subheader("Tone Applied")
                st.markdown(f"> {rewritten_text}")
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original")
                st.markdown(f"> {original_text[:150]}...")
            with col2:
                st.subheader("Tone Applied")
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

                # Create filename with processing info
                filename_parts = [selected_tone, selected_voice]
                if use_summarizer:
                    filename_parts.insert(0, "Summarized")
                filename = f"EchoVerse_{'_'.join(filename_parts)}.mp3"

                st.download_button(
                    label="📥 Download Audiobook",
                    data=audio_bytes,
                    file_name=filename,
                    mime="audio/mpeg"
                )

                # Store in session state
                st.session_state.narrations.append({
                    "original_text": original_text,
                    "final_text": final_text,
                    "rewritten_text": rewritten_text,
                    "tone": selected_tone,
                    "voice": selected_voice,
                    "summarized": use_summarizer,
                    "audio_bytes": audio_bytes,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                })

                st.success("✅ Audiobook generated successfully!")
                
            except Exception as e:
                st.error(f"TTS Error: {e}")

# --- Past Narrations ---
if st.session_state.narrations:
    st.divider()
    with st.expander("📚 Past Narrations History"):
        for i, narration in enumerate(reversed(st.session_state.narrations)):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                processing_info = []
                if narration.get('summarized', False):
                    processing_info.append("📝 Summarized")
                processing_info.extend([f"🎭 {narration['tone']}", f"🎤 {narration['voice']}"])
                
                st.markdown(f"**Narration {len(st.session_state.narrations)-i}** | {' | '.join(processing_info)}")
                st.caption(f"Created: {narration.get('timestamp', 'Unknown')}")
                
                # Show text preview
                preview_text = narration.get('final_text', narration['rewritten_text'])
                st.markdown(f"> {preview_text[:200]}{'...' if len(preview_text) > 200 else ''}")
                
                st.audio(narration['audio_bytes'], sample_rate=16000)
            
            with col2:
                filename_parts = [narration['tone'], narration['voice']]
                if narration.get('summarized', False):
                    filename_parts.insert(0, "Summarized")
                filename = f"EchoVerse_{'_'.join(filename_parts)}.mp3"
                
                st.download_button(
                    label="📥 Re-download",
                    data=narration['audio_bytes'],
                    file_name=filename,
                    mime="audio/mpeg",
                    key=f"download_{i}"
                )
            
            if i < len(st.session_state.narrations) - 1:
                st.markdown("---")

# --- Footer ---
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit and Microsoft SpeechT5")

