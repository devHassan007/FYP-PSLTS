# SignLink — Real-Time PSL Sign Language Translator

A Chrome extension that detects Pakistani Sign Language during 
video calls and shows real-time urdu-subtitles with English Speech.

##  Website
https://devHassan007.github.io/SignLink


##  Model Performance
- Accuracy: 88.3% ± 3.38%
- Test Accuracy: 100% (17/17)
- Architecture: Bidirectional LSTM (hidden=64, layers=2)
- Training: MediaPipe landmarks → normalized sequences → BiLSTM

##  Quick Start

### 1. Clone repo
git clone https://github.com/devHassan007/SignLink.git
cd SignLink

### 2. Install dependencies
pip install -r requirements.txt

### 3. Start backend server
python backend/server.py

### 4. Load Chrome extension
Open chrome://extensions → Developer Mode ON
→ Load Unpacked → select /extension folder

### 5. Open Google Meet
SignLink auto-detects signing and shows subtitles.

##  Architecture
Browser Extension → WebSocket → FastAPI Server
→ MediaPipe (225 landmarks) → BiLSTM Model → Prediction

##  Project Structure
- /backend      FastAPI WebSocket server
- /extension    Chrome extension files
- /scripts      Training and dataset scripts
- /models       Trained model weights (.pth)
- /website      Landing page
- /psl_dataset  Dataset labels

##  Tech Stack
- PyTorch (BiLSTM model)
- MediaPipe Holistic (landmark extraction)
- FastAPI + WebSocket (backend)
- Chrome Extension Manifest V3
