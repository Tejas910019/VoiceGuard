import streamlit as st
import numpy as np
import librosa
import tempfile
import os
import subprocess

st.title("VoiceGuard Prototype")
st.caption("AI Voice Clone Detection")

uploaded_file = st.file_uploader("Upload audio (.wav or .mp3)")

def load_audio_robust(path, sr_target=16000):
    # Use ffmpeg to convert to WAV (Works on Python 3.14)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
        wav_path = temp_wav.name
    
    command = ['ffmpeg', '-y', '-i', path, '-ar', str(sr_target), '-ac', '1', wav_path]
    subprocess.run(command, check=True, capture_output=True)

    try:
        y, sr = librosa.load(wav_path, sr=sr_target)
    except Exception:
        # Fallback if librosa fails on 3.14
        import scipy.io.wavfile as wavfile
        sr, y_int = wavfile.read(wav_path)
        y = y_int.astype(np.float32) / 32768.0
        
    os.remove(wav_path)
    return y, sr

if uploaded_file is not None:
    file_extension = os.path.splitext(uploaded_file.name)[1]

    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_extension)
    tmp_file.write(uploaded_file.getvalue())
    tmp_file.close()
    tmp_file_path = tmp_file.name

    try:
        y, sr = load_audio_robust(tmp_file_path, sr_target=16000)

        # --- NEW HEURISTIC FOR HACKATHON ---
        flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)))
        
        # DEBUG: Display the raw values so you can tune it!
        st.write(f"🔍 **Debug:** Spectral Flatness = {flatness:.4f}")
        
        # NEW FORMULA: Human voices have HIGHER flatness (more noise/breaths) 
        # AI voices have VERY LOW flatness (too clean)
        # 3000 is the multiplier. If the human scores too low, lower it to 2000. If AI scores too high, raise it to 4000.
        trust_score = int(min(100, max(0, flatness * 3000)))

        st.subheader(f"Trust Score: {trust_score} / 100")
        
        if trust_score >= 65:
            st.success("LOW RISK: Authentic Human Voice")
        elif 40 <= trust_score < 65:
            st.warning("SUSPICIOUS: Step-Up Verification Required")
        else:
            st.error("HIGH-RISK: AI Clone Detected")
            
    finally:
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)
