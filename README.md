# Saudi-Currency-Recognition
Currency recognition system with counterfeit detection, built with Python, OpenCV, and Streamlit.

# Saudi Currency Recognition & Fake Note Detection

A real-time computer vision system to recognize Saudi currency notes and detect counterfeit ones.

---

## Features
- Detects multiple notes in a single frame
- Identifies denominations using **SIFT + FLANN feature matching**
- Fake note detection via **average color deviation + edge analysis**
- Faster recognition with **descriptor caching (.pkl)**
- Streamlit app with live camera support

---

## Files
- `prepare_descriptors.py` → Extracts SIFT descriptors from dataset images and saves them in `descriptors.pkl`
- `app_demo.py` → Streamlit app for real-time recognition
- `requirements.txt` → List of required Python libraries
