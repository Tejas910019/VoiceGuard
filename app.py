import streamlit as st
import numpy as np
import librosa
import tempfile
import os
import subprocess
import scipy.signal

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
        import scipy.io.wavfile as wavfile
        sr, y_int = wavfile.read(wav_path)
        y = y_int.astype(np.float32) / 32768.0
    os.remove(wav_path)
    return y, sr

def clean_voice(y, sr):
    # High-pass filter
    sos = scipy.signal.butter(10, 80, 'hp', fs=sr, output='sos')
    y = scipy.signal.sosfilt(sos, y)
    # Pre-emphasis
    y = librosa.effects.preemphasis(y, coef=0.97)
    # Noise Shield: Separate harmonic voice from percussive noise
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    return y_harmonic

if uploaded_file is not None:
    file_extension = os.path.splitext(uploaded_file.name)[1]
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_extension)
    tmp_file.write(uploaded_file.getvalue())
    tmp_file.close()
    tmp_file_path = tmp_file.name

    try:
        y, sr = load_audio_robust(tmp_file_path, sr_target=16000)
        
        # Noise Gate (Trim Silence)
        intervals = librosa.effects.split(y, top_db=30)
        if len(intervals) > 0:
            y = y[intervals[0][0]:intervals[-1][1]]

        # FIXED NORMALIZATION (Replaces broken librosa.effects.normalize)
        max_val = np.max(np.abs(y))
        if max_val > 0:
            y = y / max_val

        # Isolate the voice
        y_voice = clean_voice(y, sr)

        # FEATURE 1: Spectral Contrast Variance
        # Humans have sharp peaks and valleys in frequency. AI is extremely smooth.
        contrast = librosa.feature.spectral_contrast(y=y_voice, sr=sr)
        contrast_var = np.var(contrast)

        # FEATURE 2: Spectral Roll-off
        # AI voices lack high frequencies (band-limited). Human voices have rich highs.
        rolloff = librosa.feature.spectral_rolloff(y=y_voice, sr=sr, roll_percent=0.90)
        rolloff_mean = np.mean(rolloff)

        # Debug so you can tune it live
        st.write(f"🔍 **Debug Contrast Var:** {contrast_var:.4f}")
        st.write(f"🔍 **Debug Roll-off Mean:** {rolloff_mean:.2f}")

        # The "Judges Killer" Slider
        sensitivity = st.slider("AI Detection Sensitivity", 10, 200, 60)

        # Final Trust Score (Log1p prevents crazy 100/100 scores)
        raw_score = (np.log1p(contrast_var) * 80) + (np.log1p(rolloff_mean) * 10)
        trust_score = int(np.clip(raw_score * (sensitivity / 60), 0, 100))

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
