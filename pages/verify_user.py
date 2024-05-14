import os.path
import sys

import numpy as np
import streamlit as st
from PIL import Image
import os


from scripts import VoiceRecognitionSystem

st.set_page_config(page_title="Verify User", page_icon="🎤️")


def verify_user(user_name: str, user_voice: bytes) -> bool:
    column_left, column_right = st.columns(2)
    try:
        is_verified, _ = voice_ver_system.verify_user(
            user_name=user_name, user_voice_path=user_voice
        )

        if is_verified:
            st.success("Verified")
            st.markdown('<style>h1{color: green;}</style>', unsafe_allow_html=True)
        else:
            st.error("Not Verified")
            st.markdown('<style>h1{color: red;}</style>', unsafe_allow_html=True)
        return is_verified
    except Exception as e:
        with column_right:
            st.exception(e)


def play_user_rec(user_name: str):
    '''play 3 random recordings of the user'''
    raise NotImplementedError


st.title("Verify User")
voice_ver_system = VoiceRecognitionSystem(
    database_path=os.path.join("data", "database")
)

st.header("Please enter your username and upload your voice")
uploaded_name = st.text_input(label="Username").lower()

if uploaded_name:
    uploaded_voice = st.file_uploader(label="Upload Voice", type=["wav"])
    if uploaded_voice is not None:
        st.audio(uploaded_voice, format='audio/wav')

        is_verified = verify_user(user_name=uploaded_name, user_voice=uploaded_voice)
        # if is_verified:
        # play_user_rec(user_name=uploaded_name)
