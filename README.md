# Saudi-Currency-Recognition
Currency recognition system a computer vision project in my junior year focusing on counterfeit detection, built with Python, OpenCV, and Streamlit.

<img width="691" height="385" alt="image" src="https://github.com/user-attachments/assets/af2dad24-5876-4709-9343-a9384be2a1e8" />



# Saudi Currency Recognition & Fake Note Detection

Saudi Currency Recognition is a real-time computer vision system designed to recognize Saudi currency notes and detect counterfeit ones. This project was developed as part of our Computer Vision course by Reem, Rimas,and Ibtihal.

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
- `requirements.txt` → List of required Python libraries, Dataset and steps to run the project
