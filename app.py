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

# Noise Shield
def clean_voice(y, sr):
    sos = scipy.signal.butter(10, 80, 'hp', fs=sr, output='sos')
    y = scipy.signal.sosfilt(sos, y)
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
        
        # Noise Gate + Normalize
        intervals = librosa.effects.split(y, top_db=30)
        if len(intervals) > 0:
            y = y[intervals[0][0]:intervals[-1][1]]
        max_val = np.max(np.abs(y))
        if max_val > 0: y = y / max_val

        y_voice = clean_voice(y, sr)

        # --- THE PITCH JITTER DETECTOR ---
        # We extract the fundamental frequency ONLY when voice is active (voiced_flag)
        # Noise has no pitch, so it gets filtered out by the mask.
        f0, voiced_flag, voiced_prob = librosa.pyin(
            y_voice, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'),
            sr=sr, frame_length=2048
        )
        
        # Filter out the nan values (silence)
        valid_pitch = f0[~np.isnan(f0)]
        
        if len(valid_pitch) == 0:
            st.error("Could not detect a strong voice signal. Please try a clearer recording.")
        else:
            # Calculate the Standard Deviation (Jitter)
            pitch_std = np.std(valid_pitch)

            # DEBUG FOR TUNING
            st.write(f"🔍 **Pitch Jitter (Std Dev):** {pitch_std:.2f}")

            # TUNING SLIDER (Adjust this to make AI fail and Human pass)
            sensitivity = st.slider("AI Detection Sensitivity", 1, 20, 6)

            # Formula: AI = very low (1-3), Human = high (10-30)
            trust_score = int(np.clip((pitch_std * sensitivity), 0, 100))

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
