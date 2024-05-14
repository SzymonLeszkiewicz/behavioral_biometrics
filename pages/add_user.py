import os.path
import os
import streamlit as st

from scripts import VoiceRecognitionSystem

st.set_page_config(page_title="Add User", page_icon="🎤️")


def add_user_voice(user_name, user_voice):
    user_directory_path = os.path.join(
                                       "data", "database", "authorized_users", user_name
                                       )
    if not os.path.exists(user_directory_path):
        os.makedirs(user_directory_path, exist_ok=True)

    path = os.path.join(user_directory_path, user_voice.name)
    with open(path, "wb") as f:
        f.write(user_voice.read())


st.title("Add User")
st.header("Please enter your username and upload your voice")
face_verification_system = VoiceRecognitionSystem(
    database_path=os.path.join( "data", "database"))

uploaded_name = st.text_input(label="Username").lower()

if uploaded_name:
    user_directory_path = os.path.join(
                                       "data", "database", "authorized_users", uploaded_name
                                       )
    if os.path.exists(user_directory_path):
        t = "welcome back 👋"
        st.write(f"### :green[ {t} ]")
    else:
        t = "Create profile by uploading images"
        st.write(f"### :yellow[ {t} ]")

    uploaded_voice = st.file_uploader(label="Upload Voice", type=["wav"])

    if uploaded_voice:
        st.audio(uploaded_voice, format='audio/wav')
        add_user_voice(user_name=uploaded_name, user_voice=uploaded_voice)
