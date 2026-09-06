import streamlit as st
import numpy as np
import librosa
import tempfile
import os
import soundfile as sf
import audioread

st.title("VoiceGuard Prototype")
st.caption("AI Voice Clone Detection")

uploaded_file = st.file_uploader("Upload audio (.wav or .mp3)")

# Custom robust audio loader
def load_audio_robust(path, sr_target=16000):
    """Try librosa first, then fallback to audioread (for MP3 and corrupted files)."""
    try:
        # Try standard librosa/soundfile loader (handles WAV, FLAC well)
        y, sr = librosa.load(path, sr=sr_target)
    except Exception:
        # Manual fallback to audioread (uses ffmpeg for MP3)
        with audioread.audio_open(path) as f:
            sr = f.samplerate
            # Convert int16 buffer to float32 (required by librosa)
            y = np.frombuffer(f.read(), dtype=np.int16).astype(np.float32) / 32768.0
        
        # Resample to target sample rate if different
        if sr != sr_target:
            y = librosa.resample(y, orig_sr=sr, target_sr=sr_target)
            sr = sr_target
            
    return y, sr

if uploaded_file is not None:
    file_extension = os.path.splitext(uploaded_file.name)[1]

    # Create the temp file, write data, and explicitly close it
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_extension)
    tmp_file.write(uploaded_file.getvalue())
    tmp_file.close()
    tmp_file_path = tmp_file.name

    try:
        # Use the robust loader instead of librosa.load directly
        y, sr = load_audio_robust(tmp_file_path, sr_target=16000)
        
        flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)))
        trust_score = max(0, min(100, 100 - int(flatness * 5000)))

        st.subheader(f"Trust Score: {trust_score} / 100")
        
        if trust_score >= 65:
            st.success("LOW RISK: Authentic Human Voice")
        elif 40 <= trust_score < 65:
            st.warning("SUSPICIOUS: Step-Up Verification Required")
        else:
            st.error("HIGH-RISK: AI Clone Detected")
            
    finally:
        # Safely cleanup the temp file
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)
