import streamlit as st
import pickle
import time
from rsm_parser import ext_txt_from_pdf, ext_rsm_dtl
from feat_ext import ext_feats


with open("resume_ranker.pkl", "rb") as file:
    model, vectorizer = pickle.load(file)

theme_css = """
    <style>
        .title { font-size: 36px; font-weight: bold; text-align: center; color: #4CAF50; }
        .subheader { font-size: 20px; text-align: center; color: #FFFFFF; }
        .container { background-color: #1E1E1E; padding: 20px; border-radius: 10px; }
        .result { font-size: 24px; font-weight: bold; text-align: center; color: #FFD700; }
    </style>
"""
st.markdown(theme_css, unsafe_allow_html=True)

st.markdown('<p class="title"> AI Resume Screening System</p>', unsafe_allow_html=True)
st.markdown('<p class="subheader">Upload a resume and get instant candidate ranking</p>', unsafe_allow_html=True)

st.markdown("### Upload Resume (PDF)")
uploaded_file = st.file_uploader("", type=["pdf"])

if uploaded_file:
    st.success("Resume uploaded successfully!")
    
    resume_text = ext_txt_from_pdf(uploaded_file)
    details = ext_rsm_dtl(resume_text)
    
    st.markdown("### Extracted Resume Details")
    st.json(details)
    
    st.markdown("### Analyzing Resume...")
    progress_bar = st.progress(0)
    for percent in range(100):
        time.sleep(0.02)
        progress_bar.progress(percent + 1)
    
    X_resume = vectorizer.transform([resume_text])
    ranking = model.predict_proba(X_resume)[0, 1]
    
    st.markdown(f'<p class="result"> Candidate Ranking Score: {ranking:.2f}</p>', unsafe_allow_html=True)
