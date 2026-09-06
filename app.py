import streamlit as st
import numpy as np
import librosa
import tempfile
import os
import pydub  # Import pydub

st.title("VoiceGuard Prototype")
st.caption("AI Voice Clone Detection")

uploaded_file = st.file_uploader("Upload audio (.wav or .mp3)")

def load_audio_robust(path, sr_target=16000):
    # 1. Try loading directly (works perfectly for .wav files)
    try:
        y, sr = librosa.load(path, sr=sr_target)
        return y, sr
    except Exception:
        # 2. Fallback: Use pydub to convert MP3/other formats to WAV first
        try:
            # Read the audio using ffmpeg (pydub)
            audio = pydub.AudioSegment.from_file(path)
            
            # Export it as a temporary clean .wav file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
                audio.export(temp_wav.name, format="wav")
                wav_path = temp_wav.name
            
            # Load the clean WAV file with librosa
            y, sr = librosa.load(wav_path, sr=sr_target)
            
            # Delete the temporary WAV file
            os.remove(wav_path)
            return y, sr
            
        except Exception as e:
            raise RuntimeError(f"Could not process the audio: {e}")

if uploaded_file is not None:
    file_extension = os.path.splitext(uploaded_file.name)[1]

    # Create the temp file
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_extension)
    tmp_file.write(uploaded_file.getvalue())
    tmp_file.close()
    tmp_file_path = tmp_file.name

    try:
        # Use the robust loader
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
