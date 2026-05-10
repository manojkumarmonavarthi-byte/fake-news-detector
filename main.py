import streamlit as st
import pickle
import pandas as pd
import plotly.express as px

from database import *

st.set_page_config(
    page_title="Fake News Detector",
    layout="centered"
)

create_tables()

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# ---------------- SESSION ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = ""

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

        if username.strip() == "" or password.strip() == "":
            st.warning("Please enter all details")

        else:

            result = add_user(username, password)

            if result:
                st.success("Account Created Successfully")

            else:
                st.error("Username Already Exists")

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

        result = login_user(username, password)

        if result:

            st.session_state.logged_in = True
            st.session_state.username = username

            st.rerun()

        else:
            st.error("Invalid Username or Password")

# ---------------- MAIN APP ----------------
def app():

    st.title("📰 Fake News Detector")

    st.success(f"Welcome {st.session_state.username}")

    news = st.text_area("Enter News Text")

    if st.button("Predict"):

        if news.strip() != "":

            # Transform input
            transformed_news = vectorizer.transform([news])

            # Prediction
            prediction = model.predict(transformed_news)[0]

            # Confidence score
            confidence_score = model.decision_function(
                transformed_news
            )

            confidence = round(
                abs(confidence_score[0]) * 10,
                2
            )

            # Result
            if prediction == 0:

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

            # Save history
            add_history(
                st.session_state.username,
                news,
                result
            )

        else:
            st.warning("Please enter news text")

    # ---------------- HISTORY ----------------
    history = get_history(
        st.session_state.username
    )

    st.subheader("📜 Prediction History")

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
            title="Fake vs Real Predictions"
        )

        st.plotly_chart(fig)

    else:
        st.warning(
            "No Prediction History Found"
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

    menu = st.sidebar.selectbox(
        "Menu",
        ["Login", "Signup"]
    )

    if menu == "Login":
        login()

    else:
        signup()