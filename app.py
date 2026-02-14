import streamlit as st
import pandas as pd
import pickle

# ---------------- TRAINED MODEL ----------------
model = pickle.load(open("insurance_model.pk1", "rb"))

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Insurance App", layout="wide")

# ---------------- SIDEBAR NAVIGATION ----------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Prediction Page", "About Page"])

# =====================================================
# ---------------- PAGE 1 : INPUT & OUTPUT ------------
# =====================================================
if page == "Prediction Page":

    st.title("🏥 Insurance Prediction")

    st.subheader("Enter Details")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=5, max_value=100)
        sex = st.selectbox("Sex", ["male", "female"])
        bmi = st.number_input("BMI", min_value=5.0, max_value=80.0)

    with col2:
        children = st.number_input("Number of Children", min_value=0, max_value=5)
        smoker = st.selectbox("Smoker", ["yes", "no"])
        region = st.selectbox(
            "Region",
            ["northeast", "northwest", "southeast", "southwest"]
        )

    if st.button("Submit"):
        
        # You will add ML code here
        result = "Your Input Is Not Correct"
        user_input = pd.DataFrame([[age, sex, bmi, children, smoker, region]],
                  columns=['age','sex','bmi','children','smoker','region'])
        result = model.predict(user_input)
        st.subheader("Output")
        st.success(result.item())


# =====================================================
# ---------------- PAGE 2 : ABOUT ---------------------
# =====================================================
elif page == "About Page":

    st.title("👨‍💻 About Me")

    st.markdown("""
    **Name:** Priyansi Yadav\n
    **GitHub ID:** https://github.com/priyanshiyadav03/
    """)
    st.markdown("""
    **Name:** Pranjali Singh\n
    **GitHub ID:** https://github.com/pranjali-t 
    """)
    st.markdown("""
    **Name:** Naitik Singh\n
    **GitHub ID:** https://github.com/Naitik152singh 
    """)
    st.markdown("""
    **Name:** Nikhil Kumar\n
    **GitHub ID:** https://github.com/Nikhil-Kumar-2007
    """)
    st.markdown("""
    **Project Repository:** https://github.com/priyanshiyadav03/Regression-Analysis
    """)