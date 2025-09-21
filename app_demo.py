import os
import cv2
import numpy as np
import streamlit as st
from PIL import Image
import pickle
import time
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

# ======= Average Color Deviation (Simple Fake Check) =======
def calculate_color_deviation(note_image, real_avg_color=(150, 150, 150), threshold=80):
    # Compute the average color of the note
    avg_color = np.mean(note_image, axis=(0, 1))  # Mean across the height and width
    color_diff = np.linalg.norm(avg_color - np.array(real_avg_color))  # Euclidean distance between avg color and real color
    
    # If the color difference is larger than the threshold, it's a fake note
    return color_diff > threshold

# ======= Feature Extraction =======
def extract_features(image):
    sift = cv2.SIFT_create(nfeatures=5000)
    _, descriptors = sift.detectAndCompute(image, None)
    return descriptors

# ======= Feature Matching =======
def match_features(desc1, desc2):
    index_params = dict(algorithm=1, trees=10)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(desc1, desc2, k=2)
    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]
    return len(good_matches)

# ======= Currency Identification (Parallel) =======
def identify_currency(frame, reference_data, min_score=15):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    desc1 = extract_features(gray)
    best_matches = {}

    ref_pairs = []
    for label, descriptors_list in reference_data.items():
        for data in descriptors_list:
            ref_pairs.append((label, data['desc']))

    def match_task(pair):
        label, desc2 = pair
        if desc1 is not None and desc2 is not None:
            score = match_features(desc1, desc2)
            return (label, score)
        return (label, 0)

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(match_task, ref_pairs))

    for label, score in results:
        if score >= min_score:
            if label not in best_matches or score > best_matches[label][1]:
                best_matches[label] = (label, score)

    filtered_results = list(best_matches.values())
    if filtered_results:
        return max(filtered_results, key=lambda x: x[1])
    return None

# ======= Currency Note Detection =======
def detect_currency_notes(frame, max_notes=5):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blur, 30, 150)
    edged = cv2.dilate(edged, None, iterations=2)

    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    note_images = []

    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for cnt in contours[:max_notes]:
        area = cv2.contourArea(cnt)
        if area < 5000:
            continue
        if area > frame.shape[0] * frame.shape[1] * 0.9:
            continue  # Skip large contour (the full image)
        x, y, w, h = cv2.boundingRect(cnt)
        cropped = frame[y:y+h, x:x+w]
        note_images.append(cropped)

    return note_images

# ======= Load Cached Descriptors =======
@st.cache_resource
def load_cached_descriptors():
    with open("descriptors.pkl", "rb") as f:
        return pickle.load(f)

# ======= Currency Values =======
currency_values = {
    "1 SR": 1,
    "5 SR": 5,
    "10 SR": 10,
    "50 SR": 50,
    "100 SR": 100,
    "500 SR": 500
}

usd_rate = 0.27
eur_rate = 0.24

# ======= Streamlit App =======
st.set_page_config(page_title="Saudi Currency Recognition", layout="centered")
st.title("💵 Saudi Currency Recognition (Live Camera)")
st.write("📷 Point your camera at Saudi currency notes and press **Capture**.")

# Load reference data
ref_data = load_cached_descriptors()

# Session state
if 'results' not in st.session_state:
    st.session_state.results = []

# Camera Input
uploaded_image = st.camera_input("🎥 Live Camera Feed")

# Function to detect fake notes based on color deviation
def detect_fake_note(note_image):
    # Simple fake check based on average color deviation
    real_avg_color = (150, 150, 150)  # Typical average color for real currency notes
    if calculate_color_deviation(note_image, real_avg_color):
        return "❌ Fake"
    return "✅ Real"

if uploaded_image and st.button("📷 Capture Currency"):
    img = Image.open(uploaded_image)
    img_array = np.array(img.convert("RGB"))
    frame = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    cropped_notes = detect_currency_notes(frame)
    st.info(f"🧠 Detected {len(cropped_notes)} note(s)")

    if not cropped_notes:
        st.warning("⚠️ No currency notes detected. Try again with a clearer frame.")
    else:
        for note in cropped_notes:
            start_time = time.time()
            result = identify_currency(note, ref_data)
            duration = time.time() - start_time

            st.image(note, caption=f"Detected Note - ⏱ {duration:.2f}s", channels="BGR")

            # Fake note detection based on color deviation
            verdict = detect_fake_note(note)
            st.text(f"Authenticity: {verdict}")

            if verdict == "✅ Real":
                if isinstance(result, tuple):
                    label = result[0]
                    value = currency_values.get(label, 0)
                    st.session_state.results.append({
                        "label": label,
                        "matches": result[1],
                        "value": value
                    })
                    st.text(f"Currency: {label} | Matches: {result[1]} | Value: {value} SAR")
                else:
                    st.text("❌ No confident match found.")
            else:
                st.warning("⚠️ Fake note detected. Currency information is not shown.")

# Show Summary Button
if st.button("💱 Show Detected Currency Summary") and st.session_state.results:
    df = pd.DataFrame(st.session_state.results)

    summary = df.groupby("label").agg(
        count=("label", "count"),
        total_sar=("value", "sum")
    ).reset_index()

    summary["USD"] = summary["total_sar"] * usd_rate
    summary["EUR"] = summary["total_sar"] * eur_rate

    st.subheader("🧾 Currency Breakdown")
    st.dataframe(summary.rename(columns={
        "label": "Currency",
        "count": "Count",
        "total_sar": "Total (SAR)"
    }), use_container_width=True)

    total_sar = summary["total_sar"].sum()
    total_usd = total_sar * usd_rate
    total_eur = total_sar * eur_rate

    st.success(f"💰 Total: {total_sar} SAR | {total_usd:.2f} USD | {total_eur:.2f} EUR")

# Reset
if st.button("🔄 Restart"):
    st.session_state.results = []
    st.success("🔄 Reset complete. Ready for new captures.")
