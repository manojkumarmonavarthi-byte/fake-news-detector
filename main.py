import streamlit as st
import joblib
import pandas as pd
import plotly.express as px
import speech_recognition as sr

from streamlit_option_menu import option_menu
from database import *

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

.main {
    background-color: #0E1117;
    color: white;
}

.stButton>button {
    width: 100%;
    border-radius: 10px;
    height: 3em;
    font-size: 18px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- DATABASE ----------------
create_tables()

# ---------------- LOAD MODEL ----------------


model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
# ---------------- SESSION ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = ""

# ---------------- VOICE FUNCTION ----------------
def voice_to_text():

    recognizer = sr.Recognizer()

    try:

        with sr.Microphone() as source:

            st.warning("🎤 Listening... Speak Now")

            recognizer.adjust_for_ambient_noise(
                source,
                duration=1
            )

            audio = recognizer.listen(
                source,
                timeout=10,
                phrase_time_limit=10
            )

            st.success("✅ Recording Completed")

            text = recognizer.recognize_google(audio)

            return text

    except sr.WaitTimeoutError:

        st.error("⏰ No Voice Detected")
        return ""

    except sr.UnknownValueError:

        st.error("❌ Could Not Understand Voice")
        return ""

    except Exception as e:

        st.error(f"Error: {e}")
        return ""

# ---------------- SIGNUP ----------------
def signup():

    st.title("📝 Signup")

    username = st.text_input(
        "Username",
        key="signup_user"
    )

    password = st.text_input(
        "Password",
        type="password",
        key="signup_pass"
    )

    if st.button("Create Account"):

        if username == "" or password == "":
            st.warning("Please Fill All Fields")

        else:

            result = add_user(
                username,
                password
            )

            if result:
                st.success(
                    "Account Created Successfully"
                )

            else:
                st.error(
                    "Username Already Exists"
                )

# ---------------- LOGIN ----------------
def login():

    st.title("🔐 Login")

    username = st.text_input(
        "Username",
        key="login_user"
    )

    password = st.text_input(
        "Password",
        type="password",
        key="login_pass"
    )

    if st.button("Login"):

        result = login_user(
            username,
            password
        )

        if result:

            st.session_state.logged_in = True
            st.session_state.username = username

            st.rerun()

        else:
            st.error(
                "Invalid Username or Password"
            )

# ---------------- PREDICT FUNCTION ----------------
def predict_news(news):

    transformed_news = vectorizer.transform([news])

    prediction = model.predict(
        transformed_news
    )[0]

    probability = model.predict_proba(
        transformed_news
    )

    confidence = round(
        max(probability[0]) * 100,
        2
    )

    if prediction == 1:

        result = "REAL"

        st.success(
            f"Prediction: {result}"
        )

    else:

        result = "FAKE"

        st.error(
            f"Prediction: {result}"
        )

    st.info(
        f"Confidence Score: {confidence}%"
    )

    add_history(
        st.session_state.username,
        news,
        result
    )

# ---------------- MAIN APP ----------------
def app():

    st.title("📰 AI Fake News Detector")

    st.success(
        f"Welcome {st.session_state.username}"
    )

    selected = option_menu(
        menu_title=None,
        options=[
            "Text Prediction",
            "Voice Prediction"
        ],
        icons=[
            "file-text",
            "mic"
        ],
        orientation="horizontal"
    )

    # ---------------- TEXT PREDICTION ----------------
    if selected == "Text Prediction":

        news = st.text_area(
            "Enter News Text"
        )

        if st.button("Predict News"):

            if news != "":
                predict_news(news)

            else:
                st.warning(
                    "Please Enter News Text"
                )

    # ---------------- VOICE PREDICTION ----------------
    if selected == "Voice Prediction":

        st.subheader("🎤 Voice News Detection")

        if "voice_text" not in st.session_state:
            st.session_state.voice_text = ""

        if "recording" not in st.session_state:
            st.session_state.recording = False

        col1, col2 = st.columns(2)

        with col1:

            if st.button("🎙️ Start Recording"):

                st.session_state.recording = True

                voice_text = voice_to_text()

                if voice_text != "":

                    st.session_state.voice_text = voice_text

                    st.success(
                        "✅ Voice Recognized Successfully"
                    )

                    predict_news(voice_text)

                else:

                    st.error(
                        "❌ Could Not Recognize Voice"
                    )

        with col2:

            if st.button("⛔ Stop Recording"):

                st.session_state.recording = False

                st.info("🛑 Recording Stopped")

        if st.session_state.voice_text != "":

            st.text_area(
                "Recognized Text",
                st.session_state.voice_text,
                height=150
            )

    # ---------------- HISTORY ----------------
    st.subheader("📜 Prediction History")

    history = get_history(
        st.session_state.username
    )

    if history:

        history_df = pd.DataFrame(
            history,
            columns=["News", "Result"]
        )

        st.dataframe(history_df)

        counts = history_df[
            "Result"
        ].value_counts()

        fig = px.pie(
            values=counts.values,
            names=counts.index,
            title="Prediction Statistics"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:
        st.warning(
            "No History Found"
        )

    # ---------------- LOGOUT ----------------
    if st.button("Logout"):

        st.session_state.logged_in = False
        st.session_state.username = ""

        st.rerun()

# ---------------- MENU ----------------
if st.session_state.logged_in:

    app()

else:

    choice = st.sidebar.selectbox(
        "Menu",
        ["Login", "Signup"]
    )

    if choice == "Login":
        login()

    else:
        signup()