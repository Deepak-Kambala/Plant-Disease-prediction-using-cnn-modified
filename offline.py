import streamlit as st
import streamlit.components.v1 as components
import tensorflow as tf
import numpy as np
import pandas as pd
import folium
from streamlit_folium import st_folium
import tempfile
from datetime import datetime
import requests
import json
from offline_translations import OFFLINE_TRANSLATIONS, OFFLINE_DATA

# ------------------------------
# Try to import Google Generative AI
# ------------------------------
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

# ------------------------------
# Internet connectivity check
# ------------------------------
def check_internet_connection():
    try:
        requests.get("https://www.google.com", timeout=5)
        return True
    except requests.RequestException:
        return False

# ------------------------------
# Initialize Google GenAI client
# ------------------------------
def initialize_genai_client():
    if GENAI_AVAILABLE and check_internet_connection():
        try:
            genai.configure(api_key=st.secrets["GEMINI_API_KEY"])  
            model = genai.GenerativeModel("gemini-1.5-flash")
            return model
        except Exception:
            return None
    return None

# ------------------------------
# TensorFlow Model Prediction
# ------------------------------
def model_prediction(test_image_path):
    model = tf.keras.models.load_model('trained_model.h5', compile=False)
    image = tf.keras.preprocessing.image.load_img(test_image_path, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    prediction = model.predict(input_arr)
    return np.argmax(prediction)

# ------------------------------
# Offline translation and precautions
# ------------------------------
def get_offline_translation(disease, language):
    if language in OFFLINE_TRANSLATIONS and disease in OFFLINE_TRANSLATIONS[language]:
        return OFFLINE_TRANSLATIONS[language][disease]
    return disease

def get_offline_precautions(disease, language):
    if disease in OFFLINE_DATA and language in OFFLINE_DATA[disease]:
        return OFFLINE_DATA[disease][language]
    return {
        "precaution": "Maintain proper plant hygiene, ensure adequate spacing, and monitor regularly for signs of disease.",
        "medicine": "Use recommended fungicides or insecticides as per agricultural guidelines."
    }

# ------------------------------
# Initialize session state
# ------------------------------
for key in ["uploaded_file", "predicted_disease", "translated_prediction",
            "chat_messages", "processing_message", "preferred_language", "language_changed",
            "is_online", "genai_client", "offline_precautions_shown"]:
    if key not in st.session_state:
        if key == "chat_messages":
            st.session_state[key] = []
        elif key in ["processing_message", "language_changed", "offline_precautions_shown"]:
            st.session_state[key] = False
        elif key == "is_online":
            st.session_state[key] = check_internet_connection()
        elif key == "genai_client":
            st.session_state[key] = initialize_genai_client()
        else:
            st.session_state[key] = None

# ------------------------------
# Display connection status
# ------------------------------
if st.session_state.is_online:
    st.success("🌐 Online Mode: Full chat functionality available")
else:
    st.warning("📴 Offline Mode: Basic precautions, medicines, and translations available")

# ------------------------------
# Sidebar
# ------------------------------
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition", "Global Disease Map"])

# Refresh connection button
if st.sidebar.button("🔄 Check Connection"):
    st.session_state.is_online = check_internet_connection()
    st.session_state.genai_client = initialize_genai_client()
    st.experimental_rerun()

# ------------------------------
# Global Plant Disease Map Page
# ------------------------------
if app_mode == "Global Disease Map":
    st.header("🌾 Interactive Global Plant Disease Spread Map")
    st.markdown("""
    Check the most recent common spreading disease in your area by hovering over or clicking on the markers.
    Use the layer control (top-right corner) to change map view.
    """)

    df = pd.read_csv("global_plant_disease_data.csv")
    map_type = st.sidebar.selectbox("Select Map Type", ["Default", "OpenStreetMap"], key="map_type")
    tiles_dict = {"Default": "CartoDB positron", "OpenStreetMap": "OpenStreetMap"}
    m = folium.Map(location=[20, 0], zoom_start=2, tiles=tiles_dict[map_type])

    low_spread = folium.FeatureGroup(name="Low Spread", show=True)
    medium_spread = folium.FeatureGroup(name="Medium Spread", show=True)
    high_spread = folium.FeatureGroup(name="High Spread", show=True)

    for _, row in df.iterrows():
        color = "green" if row["spread_level"] == "Low" else "orange" if row["spread_level"] == "Medium" else "red"
        marker = folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=8,
            color=color,
            fill=True,
            fill_opacity=0.7,
            popup=folium.Popup(
                f"<b>{row['state']}, {row['country']}</b><br>🌿 <b>Crop:</b> {row['plant']}<br>🦠 <b>Disease:</b> {row['disease']}<br>📈 <b>Spread:</b> {row['spread_level']}",
                max_width=300
            ),
            tooltip=f"{row['state']}, {row['country']} — {row['disease']}"
        )
        if row["spread_level"] == "Low":
            low_spread.add_child(marker)
        elif row["spread_level"] == "Medium":
            medium_spread.add_child(marker)
        else:
            high_spread.add_child(marker)

    low_spread.add_to(m)
    medium_spread.add_to(m)
    high_spread.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)

    legend_html = """
    <div style="
        position: fixed; 
        bottom: 50px; left: 50px; width: 220px; height: 120px; 
        z-index:9999; 
        background-color:white;
        border:2px solid grey; 
        border-radius:6px; 
        padding: 10px;
        box-shadow: 3px 3px 6px rgba(0,0,0,0.2);
        font-size:14px;
    ">
    <b>Spread Level Legend</b><br>
    <div style="color:red;">🟥 High Spread</div>
    <div style="color:orange;">🟠 Medium Spread</div>
    <div style="color:green;">🟢 Low Spread</div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    st_folium(m, width=1200, height=700)

# ------------------------------
# Home Page
# ------------------------------
elif app_mode == "Home":
    st.header("🌿 Plant Disease Recognition System")
    st.video("https://youtu.be/Zw7dGLWzaXk")
    st.markdown("""
    Welcome to the Plant Disease Recognition System!  

    **Features:**
    - 🌐 **Online Mode:** Full chat functionality with AI assistant  
    - 📴 **Offline Mode:** Basic detection with precautions, medicines & translations
    """)

# ------------------------------
# About Page
# ------------------------------
elif app_mode == "About":
    st.header("About")
    st.markdown("""
    #### About Dataset
    Dataset: 87K RGB images (38 classes).  
    - **Train:** 70K images  
    - **Validation:** 17K images  
    - **Test:** 33 images  

    #### Features
    - **Online Mode:** Real-time chat with AI assistant, translations  
    - **Offline Mode:** Pre-loaded translations and precautions  
    """)

# ------------------------------
# Disease Recognition Page
# ------------------------------
elif app_mode == "Disease Recognition":
    st.header("🩺 Disease Recognition")
    
    # Allow both file upload and camera capture
    uploaded_file = st.file_uploader("Choose an Image:", type=["jpg", "jpeg", "png"])
    captured_image = st.camera_input("Or take a picture with your camera")
    
    # Give priority to captured image
    if captured_image:
        st.session_state.uploaded_file = captured_image
    elif uploaded_file:
        st.session_state.uploaded_file = uploaded_file

    if st.session_state.uploaded_file:
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("📷 Show Image"):
                st.image(st.session_state.uploaded_file, use_column_width=True)

        with col2:
            if st.button("🔍 Predict Disease"):
                with st.spinner("Please wait, predicting disease..."):
                    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                        tmp_file.write(st.session_state.uploaded_file.read())
                        tmp_path = tmp_file.name

                    result_index = model_prediction(tmp_path)
                    class_name = [
                        'Apple : Apple scab', 'Apple : Black rot', 'Apple : Cedar apple rust', 'Apple : healthy',
                        'Blueberry : healthy', 'Cherry : Powdery mildew', 'Cherry : healthy',
                        'Corn : Cercospora leaf spot Gray leaf spot', 'Corn : Common rust', 'Corn : Northern Leaf Blight',
                        'Corn : healthy', 'Grape : Black rot', 'Grape : Esca (Black Measles)',
                        'Grape : Leaf blight (Isariopsis Leaf Spot)', 'Grape : healthy',
                        'Orange : Huanglongbing (Citrus greening)', 'Peach : Bacterial spot', 'Peach : healthy',
                        'Pepper bell : Bacterial spot', 'Pepper bell : healthy', 'Potato : Early blight',
                        'Potato : Late blight', 'Potato : healthy', 'Raspberry : healthy', 'Soybean : healthy',
                        'Squash : Powdery mildew', 'Strawberry : Leaf scorch', 'Strawberry : healthy',
                        'Tomato : Bacterial spot', 'Tomato : Early blight', 'Tomato : Late blight', 'Tomato : Leaf Mold',
                        'Tomato : Septoria leaf spot', 'Tomato : Spider mites Two-spotted spider mite',
                        'Tomato : Target Spot', 'Tomato : Tomato Yellow Leaf Curl Virus',
                        'Tomato : Tomato mosaic virus', 'Tomato : healthy'
                    ]
                    st.session_state.predicted_disease = class_name[result_index]
                    st.session_state.translated_prediction = None
                    st.session_state.chat_messages = []
                    st.session_state.offline_precautions_shown = False


# ------------------------------
# Full code continues with language selection, translations, offline precautions, and chat interface
# ------------------------------

        if st.session_state.preferred_language is None:
            st.markdown("### 🌐 Select Your Preferred Language")
            cols = st.columns(8)
            languages = ["English", "Hindi", "Telugu", "Tamil", "Kannada", "Malayalam", "Bengali", "Punjabi"]
            flags = ["🇺🇸","🇮🇳","🇮🇳","🇮🇳","🇮🇳","🇮🇳","🇮🇳","🇮🇳"]
            for i, col in enumerate(cols):
                if col.button(f"{flags[i]} {languages[i]}"):
                    st.session_state.preferred_language = languages[i]
                    st.session_state.language_changed = True
                    st.experimental_rerun()

        else:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.info(f"🌐 Selected Language: **{st.session_state.preferred_language}**")
            with col2:
                if st.button("Change Language"):
                    st.session_state.preferred_language = None
                    st.session_state.language_changed = True
                    st.session_state.offline_precautions_shown = False
                    st.experimental_rerun()


    # ------------------------------
    # Translation
    # ------------------------------
    if st.session_state.language_changed and st.session_state.preferred_language:
        if st.session_state.is_online and st.session_state.genai_client:
            with st.spinner("Translating prediction..."):
                try:
                    prompt_pred = (
                        f"Translate this plant disease prediction into "
                        f"{st.session_state.preferred_language}: '{st.session_state.predicted_disease}'. "
                        f"Give only the translation."
                    )
                    response_pred = st.session_state.genai_client.generate_content(prompt_pred)
                    st.session_state.translated_prediction = response_pred.text.strip()
                except Exception:
                    st.session_state.translated_prediction = get_offline_translation(
                        st.session_state.predicted_disease, st.session_state.preferred_language)
        else:
            st.session_state.translated_prediction = get_offline_translation(
                st.session_state.predicted_disease, st.session_state.preferred_language)
        st.session_state.language_changed = False

    # ------------------------------
    # Show prediction
    # ------------------------------
    if st.session_state.predicted_disease and st.session_state.preferred_language:
        if st.session_state.translated_prediction:
            st.success(f"✅ Model Prediction: **{st.session_state.translated_prediction}**")
        else:
            st.success(f"✅ Model Prediction: **{st.session_state.predicted_disease}**")

    # ------------------------------
    # Offline precautions
    # ------------------------------
    if (not st.session_state.is_online and st.session_state.predicted_disease and
        st.session_state.preferred_language and not st.session_state.offline_precautions_shown):
        st.markdown("---")
        st.subheader("🛡️ Disease Precautions & Treatment")
        precautions = get_offline_precautions(st.session_state.predicted_disease, st.session_state.preferred_language)
        st.info("📴 **Offline Mode**: Showing pre-loaded precautions and medicines")
        st.write("**Recommended Precautions:**")
        st.write(precautions["precaution"])
        st.write("**Recommended Medicines:**")
        st.write(precautions["medicine"])
        st.session_state.offline_precautions_shown = True

    # ------------------------------
    # Chat interface (Online only)
    # ------------------------------
    if st.session_state.predicted_disease and st.session_state.preferred_language:
        st.markdown("---")
        st.subheader("💬 Chat with Agricultural Assistant")

        # Display chat messages
        chat_html = '<div id="chat-container" style="height:350px; overflow-y:auto; border:1px solid #ddd; padding:10px; border-radius:15px; background-color:#fafafa;">'
        for msg in st.session_state.chat_messages:
            time_str = datetime.now().strftime("%H:%M")
            if msg["role"] == "user":
                chat_html += f"""
                <div style="display:flex; justify-content:flex-end; margin:5px 0;">
                    <div style="background-color:#dcf8c6; padding:10px; border-radius:15px; max-width:70%;">
                        👨‍🌾 {msg['text']}<br><span style="font-size:10px; color:#555;">{time_str}</span>
                    </div>
                </div>"""
            else:
                chat_html += f"""
                <div style="display:flex; justify-content:flex-start; margin:5px 0;">
                    <div style="background-color:#f1f0f0; padding:10px; border-radius:15px; max-width:70%;">
                        🤖 {msg['text']}<br><span style="font-size:10px; color:#555;">{time_str}</span>
                    </div>
                </div>"""
        chat_html += '<div id="chat-end"></div></div>'
        chat_html += """
        <script>
            var chatContainer = document.getElementById("chat-container");
            var chatEnd = document.getElementById("chat-end");
            if(chatContainer && chatEnd){ chatEnd.scrollIntoView({behavior:"smooth"}); }
        </script>
        """
        components.html(chat_html, height=360)

        # Chat input
        user_input = st.text_input("Ask about precautions or treatment:", placeholder="e.g., How can I treat this disease?")
        if st.button("Send") and user_input.strip():
            st.session_state.chat_messages.append({"role": "user", "text": user_input.strip()})
            if st.session_state.is_online and st.session_state.genai_client:
                with st.spinner("Assistant is typing..."):
                    try:
                        context_prompt = (
                            f"You are an agricultural assistant. "
                            f"Reply briefly in {st.session_state.preferred_language} "
                            f"about {st.session_state.predicted_disease}."
                        )
                        messages = "\n".join(
                            [f"{m['role'].capitalize()}: {m['text']}" for m in st.session_state.chat_messages]
                        )
                        final_prompt = f"{context_prompt}\n\nConversation so far:\n{messages}\n\nAssistant:"
                        response = st.session_state.genai_client.generate_content(final_prompt)
                        reply = response.text.strip()
                        st.session_state.chat_messages.append({"role": "model", "text": reply})
                    except Exception:
                        st.session_state.chat_messages.append({
                            "role": "model",
                            "text": "Sorry, I encountered an error. Please try again."
                        })
            else:
                st.session_state.chat_messages.append({
                    "role": "model",
                    "text": "💡 Chat requires internet connection and AI client."
                })
            st.experimental_rerun()

