import streamlit as st
import numpy as np
import librosa
import tempfile
import os

st.title("🛡️ VoiceGuard Prototype")
st.caption("AI Voice Clone Detection")

uploaded_file = st.file_uploader("Upload audio (.wav or .mp3)")

if uploaded_file is not None:
    file_extension = os.path.splitext(uploaded_file.name)[1]
    
    # Create the temp file, write data, and explicitly close it
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_extension)
    tmp_file.write(uploaded_file.getvalue())
    tmp_file_path = tmp_file.name
    tmp_file.close() 
    
    try:
        # audioread will now safely use ffmpeg to decode the physical MP3
        y, sr = librosa.load(tmp_file_path, sr=16000)
        flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)))
        
        trust_score = max(10, min(95, 100 - int(flatness * 5000)))
        
        st.subheader(f"Trust Score: {trust_score} / 100")
        if trust_score >= 65:
            st.success("SAFE: Authentic Human Voice")
        elif 40 <= trust_score < 65:
            st.warning("SUSPICIOUS: Step-Up Verification Required")
        else:
            st.error("HIGH-RISK: AI Clone Detected")
            
    finally:
        os.remove(tmp_file_path)
        
