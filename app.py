import streamlit as st
import numpy as np
import librosa
import librosa.effects
import tempfile
import os
import subprocess
import scipy.signal
import scipy.io.wavfile as wavfile

st.title("VoiceGuard Prototype")
st.caption("AI Voice Clone Detection")

uploaded_file = st.file_uploader("Upload audio (.wav or .mp3)")

def load_audio_robust(path, sr_target=16000):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
        wav_path = temp_wav.name
    command = ['ffmpeg', '-y', '-i', path, '-ar', str(sr_target), '-ac', '1', wav_path]
    subprocess.run(command, check=True, capture_output=True)
    try:
        y, sr = librosa.load(wav_path, sr=sr_target)
    except Exception:
        sr, y_int = wavfile.read(wav_path)
        y = y_int.astype(np.float32) / 32768.0
    os.remove(wav_path)
    return y, sr

# NOISE SHIELD FUNCTION (The Hackathon Trick)
def clean_voice(y, sr):
    # 1. High-pass filter to remove low-end hum/fans
    sos = scipy.signal.butter(10, 80, 'hp', fs=sr, output='sos')
    y = scipy.signal.sosfilt(sos, y)
    
    # 2. Pre-emphasis to boost human vocal frequencies
    y = librosa.effects.preemphasis(y, coef=0.97)
    
    # 3. Separate noise (percussive) from voice (harmonic)
    # AI clones are highly harmonic; background noise is percussive.
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    
    # We ONLY analyze the harmonic part (the voice)
    return y_harmonic

if uploaded_file is not None:
    file_extension = os.path.splitext(uploaded_file.name)[1]
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_extension)
    tmp_file.write(uploaded_file.getvalue())
    tmp_file.close()
    tmp_file_path = tmp_file.name

    try:
        y, sr = load_audio_robust(tmp_file_path, sr_target=16000)
        
        # Run the voice through the Noise Shield
        y_voice = clean_voice(y, sr)

        # Calculate variance on the CLEAN voice
        mfccs = librosa.feature.mfcc(y=y_voice, sr=sr, n_mfcc=13)
        mfcc_var = np.mean(np.std(mfccs, axis=1))
        
        zcr = librosa.feature.zero_crossing_rate(y_voice)
        zcr_var = np.var(zcr)

        # DEBUG: Show the raw numbers so you can tune it!
        st.write(f"🔍 **Clean MFCC Variance:** {mfcc_var:.4f}")
        st.write(f"🔍 **Clean ZCR Variance:** {zcr_var:.6f}")

        # TUNING: Cleaned voice tends to have lower variance. 
        # Use 50 (MFCC) and 15 (ZCR). Adjust these if needed.
        trust_score = int(np.clip((mfcc_var * 50) + (zcr_var * 15), 0, 100))

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
