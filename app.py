import streamlit as st
import numpy as np
import librosa
import tempfile
import os
import subprocess
import scipy.io.wavfile as wavfile  # Backend for 3.14 compatibility

st.title("VoiceGuard Prototype")
st.caption("AI Voice Clone Detection")

uploaded_file = st.file_uploader("Upload audio (.wav or .mp3)")

def load_audio_robust(path, sr_target=16000):
    # Step 1: Convert ANY audio to WAV using external ffmpeg (bypasses 3.14 bugs)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
        wav_path = temp_wav.name
    
    command = ['ffmpeg', '-y', '-i', path, '-ar', str(sr_target), '-ac', '1', wav_path]
    subprocess.run(command, check=True, capture_output=True)

    # Step 2: Try loading with librosa
    try:
        y, sr = librosa.load(wav_path, sr=sr_target)
        os.remove(wav_path)
        return y, sr
    except Exception:
        # Step 3: Fallback to scipy for Python 3.14 if librosa crashes
        try:
            sr, y_int = wavfile.read(wav_path)
            y = y_int.astype(np.float32) / 32768.0
            os.remove(wav_path)
            return y, sr
        except Exception as e:
            os.remove(wav_path)
            raise RuntimeError(f"Audio processing failed: {e}")

if uploaded_file is not None:
    file_extension = os.path.splitext(uploaded_file.name)[1]

    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_extension)
    tmp_file.write(uploaded_file.getvalue())
    tmp_file.close()
    tmp_file_path = tmp_file.name

    try:
        y, sr = load_audio_robust(tmp_file_path, sr_target=16000)

        # Calculate flatness using numpy only (works on 3.14)
        # If librosa is available, use it, else use numpy directly
        try:
            flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)))
        except:
            # Numpy-based flatness calculation (avoid librosa entirely if needed)
            S = np.abs(np.fft.rfft(y))
            S = S[S > 0]
            flatness = np.exp(np.mean(np.log(S))) / np.mean(S)

        trust_score = max(0, min(100, 100 - int(flatness * 5000)))

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
