import streamlit as st
import requests

API_URL = "http://localhost:8000"  # Replace with your FastAPI server URL

st.set_page_config(page_title="Sentiment Analysis App", layout="centered")

st.title("🧠 Sentiment Analysis")

# --- Input Text ---
text_input = st.text_area("Enter text for sentiment analysis:", height=150)

# --- Model Selection ---
model_option = st.selectbox("Choose model for prediction:", ["All Models", "logistic", "naive_bayes", "svm"])

if st.button("Analyze Sentiment"):

    if not text_input.strip():
        st.warning("Please enter some text.")
    else:
        # Build request payload
        payload = {"text": text_input.strip()}

        try:
            if model_option == "All Models":
                response = requests.post(f"{API_URL}/predict_all", json=payload)
            else:
                response = requests.post(f"{API_URL}/predict/{model_option}", json=payload)

            if response.status_code == 200:
                result = response.json()

                st.subheader("📊 Results:")

                if model_option == "All Models":
                    for model_name, model_data in result["model_results"].items():
                        st.markdown(f"### 🔹 Model: {model_name}")
                        st.write(f"**Prediction**: {model_data['prediction']}")
                        st.write(f"**Confidence**: {round(model_data['confidence'] * 100, 2)}%")
                        st.write("**Metrics**:")
                        st.json(model_data["metrics_on_test_data"])
                        st.markdown("---")
                else:
                    st.markdown(f"### 🔹 Model: {result['model_name']}")
                    st.write(f"**Prediction**: {result['prediction']}")
                    st.write(f"**Confidence**: {round(result['confidence'] * 100, 2)}%")
                    st.write("**Metrics**:")
                    st.json(result["metrics_on_test_data"])

            else:
                st.error(f"Error {response.status_code}: {response.json()['detail']}")

        except Exception as e:
            st.error(f"Failed to connect to API: {e}")
