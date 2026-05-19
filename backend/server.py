# SignLink Backend Server 
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import cv2
import numpy as np
import torch
import mediapipe as mp
import json
import base64
from io import BytesIO
from PIL import Image
import sys
import os
from pathlib import Path

# Add scripts directory to path
sys.path.append(str(Path(__file__).parent.parent / "scripts"))

try:
    from train_classifier import LSTMClassifier
except ImportError:
    print("Warning: Could not import LSTMClassifier.")
    class LSTMClassifier(torch.nn.Module):
        def __init__(self, input_size=225, hidden_size=64, num_classes=4, num_layers=2):
            super().__init__()
            self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers=num_layers, 
                                     batch_first=True, dropout=0.3, bidirectional=True)
            self.fc = torch.nn.Sequential(
                torch.nn.Linear(hidden_size * 2, 64),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(64, num_classes)
            )
        
        def forward(self, x):
            lstm_out, (hn, _) = self.lstm(x)
            hn = torch.cat((hn[-2], hn[-1]), dim=1)
            return self.fc(hn)

app = FastAPI(title="SignLink Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====================== CONFIG ======================
device = None
model = None
holistic = None

SEQUENCE_LENGTH = 50
MOVEMENT_THRESHOLD = 0.015      # Adjust if needed (lower = more sensitive)
MIN_SIGN_FRAMES = 10           # Minimum frames of movement to consider a sign
IDLE_FRAMES_THRESHOLD = 6    # Frames of no movement to trigger prediction

# State variables
landmark_buffer = []
movement_history = []           # Track if frame had movement
idle_counter = 0
is_signing = False

LABELS = {
    0: {"urdu": "اسلام و علیکم", "english": "AssalamuAlaikum"},
    1: {"urdu": "آپ کی بہت مہربانی ہوگی", "english": "It would be very grateful of you"},
    2: {"urdu": "آپ کا بہت شکریہ", "english": "Thanks to you"},
    3: {"urdu": "آپ سے مل کر خوشی ہوئی", "english": "Nice To Meet You"}
}
# ===================================================

def initialize_model():
    global device, model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" Using device: {device}")
    
    model = LSTMClassifier(input_size=225, hidden_size=64, num_classes=4, num_layers=2)
    
    model_path = Path(__file__).parent.parent / "models" / "classifier_best.pth"
    if model_path.exists():
        try:
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model from: {model_path}")
        except Exception as e:
            print(f"Could not load weights: {e}")
    else:
        print(f"Model not found at {model_path}")
    
    model.to(device)
    model.eval()

def initialize_mediapipe():
    global holistic
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        refine_face_landmarks=False
    )
    print("MediaPipe initialized")

def calculate_movement(landmarks):
    if len(landmark_buffer) < 2:
        return 0.0
    prev = landmark_buffer[-1]
    diff = np.abs(landmarks - prev)
    return float(np.mean(diff))

def should_predict():
    global idle_counter
    
    if len(movement_history) < MIN_SIGN_FRAMES:
        return False
    
    # Count recent idle frames
    recent_idle = sum(1 for x in movement_history[-IDLE_FRAMES_THRESHOLD:] if not x)
    
    if recent_idle >= IDLE_FRAMES_THRESHOLD and any(movement_history[-30:]):
        return True
    return False

def extract_landmarks(image):
    try:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb_image)
        landmarks = []

        # Left hand
        if results.left_hand_landmarks:
            for lm in results.left_hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
        else:
            landmarks.extend([0.0] * 63)

        # Right hand
        if results.right_hand_landmarks:
            for lm in results.right_hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
        else:
            landmarks.extend([0.0] * 63)

        # Pose
        if results.pose_landmarks:
            for lm in results.pose_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
        else:
            landmarks.extend([0.0] * 99)

        return np.array(landmarks, dtype=np.float32)
    
    except Exception as e:
        print(f"Error extracting landmarks: {e}")
        return np.zeros(225, dtype=np.float32)

def normalize_sequence(seq):
    if len(seq) == 0:
        return np.zeros((SEQUENCE_LENGTH, 225), dtype=np.float32)
    
    seq = np.array(seq)
    if len(seq) < SEQUENCE_LENGTH:
        padding = np.zeros((SEQUENCE_LENGTH - len(seq), 225))
        seq = np.vstack((seq, padding))
    else:
        seq = seq[-SEQUENCE_LENGTH:]
    
    # Normalize
    non_zero_mask = np.any(seq != 0, axis=1)
    if np.any(non_zero_mask):
        mean = np.mean(seq[non_zero_mask], axis=0)
        seq = seq - mean
        std = np.std(seq, axis=0)
        std[std == 0] = 1.0
        seq = seq / std
    
    return seq.astype(np.float32)

@app.on_event("startup")
async def startup_event():
    print("Starting SignLink Backend Server...")
    initialize_model()
    initialize_mediapipe()
    print("Server ready!")

@app.get("/")
async def root():
    return {"status": "SignLink Backend Running", "device": str(device)}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket client connected")
    
    global landmark_buffer, movement_history, idle_counter, is_signing
    landmark_buffer = []
    movement_history = []
    idle_counter = 0
    is_signing = False

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            if message["type"] == "frame":
                # Decode frame
                img_data = base64.b64decode(message["data"].split(",")[1])
                img = Image.open(BytesIO(img_data))
                frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

                landmarks = extract_landmarks(frame)
                movement = calculate_movement(landmarks)

                # Update movement history
                has_movement = movement > MOVEMENT_THRESHOLD
                movement_history.append(has_movement)
                if len(movement_history) > 60:  # keep last 60 frames
                    movement_history.pop(0)

                # Add to buffer
                landmark_buffer.append(landmarks)
                if len(landmark_buffer) > SEQUENCE_LENGTH * 2:  # allow some extra buffer
                    landmark_buffer.pop(0)

                # === Decision Logic ===
                if has_movement:
                    is_signing = True
                    idle_counter = 0
                else:
                    idle_counter += 1

                # Predict only when sign is completed (movement stopped after signing)
                if is_signing and should_predict() and len(landmark_buffer) >= SEQUENCE_LENGTH:
                    seq = normalize_sequence(landmark_buffer[-SEQUENCE_LENGTH:])
                    seq_tensor = torch.from_numpy(seq).float().unsqueeze(0).to(device)

                    with torch.no_grad():
                        output = model(seq_tensor)
                        probabilities = torch.nn.functional.softmax(output, dim=1)
                        confidence, predicted = torch.max(probabilities, 1)
                        
                        pred_class = predicted.item()
                        conf = confidence.item()

                    if conf > 0.5:
                        response = {
                            "type": "prediction",
                            "urdu": LABELS[pred_class]["urdu"],
                            "english": LABELS[pred_class]["english"],
                            "confidence": float(conf),
                            "class": pred_class
                        }
                        await websocket.send_text(json.dumps(response))
                    
                    # Reset for next sign
                    landmark_buffer = []
                    movement_history = []
                    is_signing = False
                    idle_counter = 0

                # Optional: Send low confidence / no prediction feedback
                elif not has_movement and idle_counter > 30:
                    await websocket.send_text(json.dumps({
                        "type": "prediction",
                        "urdu": "",
                        "english": "",
                        "confidence": 0.0,
                        "class": -1
                    }))

            elif message["type"] == "reset":
                landmark_buffer = []
                movement_history = []
                is_signing = False
                await websocket.send_text(json.dumps({"type": "reset_ack"}))

    except WebSocketDisconnect:
        print(" Client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("\n" + "="*70)
    print("SignLink Backend Server (Improved)")
    print("="*70)
    print(f"Running on: http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")