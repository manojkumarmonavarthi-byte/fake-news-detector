import streamlit as st
import pickle
import speech_recognition as sr

from streamlit_option_menu import option_menu
from database import *

st.set_page_config(
    page_title="AI Fake News Detector",
    page_icon="📰",
    layout="wide"
)

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

create_usertable()

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = ""

with st.sidebar:
    selected = option_menu(
        menu_title="Menu",
        options=["Login", "Signup"],
        icons=["box-arrow-in-right", "person-plus"],
        default_index=0,
    )

if selected == "Signup":

    st.title("Create Account")

    new_user = st.text_input("Username")

    new_password = st.text_input(
        "Password",
        type="password"
    )

    if st.button("Signup"):

        add_userdata(new_user, new_password)

        st.success("Account Created Successfully")

        st.info("Go to Login Menu")

elif selected == "Login":

    if not st.session_state.logged_in:

        st.title("🔐 Login")

        username = st.text_input("Username")

        password = st.text_input(
            "Password",
            type="password"
        )

        if st.button("Login"):

            result = login_user(username, password)

            if result:

                st.session_state.logged_in = True
                st.session_state.username = username

                st.rerun()

            else:
                st.error("Invalid Credentials")

    else:

        st.title("📰 AI Fake News Detector")

        st.success(
            f"Welcome {st.session_state.username}"
        )

        tab1, tab2 = st.tabs([
            "📝 Text Prediction",
            "🎤 Voice Prediction"
        ])

        with tab1:

            st.subheader("Enter News Text")

            news_text = st.text_area(
                "",
                height=180
            )

            if st.button("Predict News"):

                if news_text.strip() == "":
                    st.warning("Please Enter News Text")

                else:

                    transformed_text = vectorizer.transform(
                        [news_text]
                    )

                    prediction = model.predict(
                        transformed_text
                    )[0]

                    probability = model.predict_proba(
                        transformed_text
                    )

                    confidence = max(probability[0]) * 100

                    if prediction == 1:
                        st.success(
                            "Prediction: REAL NEWS"
                        )
                    else:
                        st.error(
                            "Prediction: FAKE NEWS"
                        )

                    st.info(
                        f"Confidence Score: {confidence:.2f}%"
                    )

        with tab2:

            st.subheader("Upload Voice File")

            audio_file = st.file_uploader(
                "Upload Audio",
                type=["wav", "mp3"]
            )

            if audio_file is not None:

                recognizer = sr.Recognizer()

                try:

                    with sr.AudioFile(audio_file) as source:
                        audio = recognizer.record(source)

                    text = recognizer.recognize_google(audio)

                    st.write("Recognized Text:")

                    st.success(text)

                    transformed_text = vectorizer.transform(
                        [text]
                    )

                    prediction = model.predict(
                        transformed_text
                    )[0]

                    probability = model.predict_proba(
                        transformed_text
                    )

                    confidence = max(probability[0]) * 100

                    if prediction == 1:
                        st.success(
                            "Prediction: REAL NEWS"
                        )
                    else:
                        st.error(
                            "Prediction: FAKE NEWS"
                        )

                    st.info(
                        f"Confidence Score: {confidence:.2f}%"
                    )

                except:
                    st.error("Could Not Recognize Voice")

        st.divider()

        st.subheader("Sample Real News")

        st.code(
            "The Indian government announced new infrastructure projects to improve transportation systems across major cities."
        )

        st.subheader("Sample Fake News")

        st.code(
            "Aliens landed in Delhi and gifted invisible flying cars to local street vendors yesterday night."
        )

        if st.button("Logout"):

            st.session_state.logged_in = False
            st.session_state.username = ""

            st.rerun()