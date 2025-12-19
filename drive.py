import os
import base64
from io import BytesIO
import numpy as np
import cv2
import socketio
import uvicorn
from fastapi import FastAPI
from PIL import Image
from keras.models import load_model

# --- Configuration ---
MODEL_PATH = './model.h5'
MAX_SPEED = 10
host = '0.0.0.0'
port = 4567

# --- Global Variables ---
model = None

# Initialize SocketIO Async Server
sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins='*')
fast_app = FastAPI()
app = socketio.ASGIApp(sio, fast_app)


def load_prediction_model(path: str):
    """
    Safely loads the Keras model.
    """
    try:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at {path}")

        loaded_model = load_model(path, compile=False)
        print(f"Model loaded successfully from {path}")
        return loaded_model
    except Exception as e:  # pylint: disable=broad-except
        print(f"Error loading model: {e}")
        return None


def img_preprocess(img: np.ndarray) -> np.ndarray:
    """
    Preprocesses the image to match the NVIDIA model architecture.
    1. Cropping: Removes sky and car hood.
    2. Color Space: RGB to YUV.
    3. Blur: Gaussian Blur to reduce noise.
    4. Resize: To 200x66 (NVIDIA standard).
    5. Normalize: Scale pixel values to 0-1.
    """
    # Crop image (remove top 60 pixels and bottom 25 pixels)
    img = img[60:135, :, :]

    # Convert to YUV (NVIDIA model recommendation)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)

    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img / 255.0
    return img


@fast_app.get('/')
async def read_root():
    return {"status": "Autonomous Driving Server Running"}


@sio.event
async def connect(sid, _environ):
    print(f'Client connected: {sid}')


@sio.event
async def disconnect(sid):
    print(f'Client disconnected: {sid}')


@sio.on('telemetry')
async def telemetry(sid, data):
    if data:
        speed = float(data['speed'])
        try:
            image_data = base64.b64decode(data['image'])
            image = Image.open(BytesIO(image_data))
            image_array = np.asarray(image)
            processed_image = img_preprocess(image_array)

            # Add batch dimension: (200, 66, 3) -> (1, 200, 66, 3)
            batch_image = np.array([processed_image])

            if model:
                steering_angle = float(model.predict(batch_image, verbose=0))

                # Reduce throttle as speed increases to prevent overspeeding
                throttle = 1.0 - (speed / MAX_SPEED)

                # (Simple version: just ensure it doesn't go too low if we want constant movement)
                throttle = max(0.1, throttle)

                print(
                    f'Steering: {steering_angle:.4f} | Throttle: {throttle:.2f} | Speed: {speed:.1f}')

                await send_control(sid, steering_angle, throttle)
            else:
                # Fallback if model isn't loaded
                await send_control(sid, 0, 0)

        except Exception as e:  # pylint: disable=broad-except
            print(f"Error processing telemetry: {e}")
            await send_control(sid, 0, 0)

    else:
        # Manual mode (usually not sent by Udacity sim, but good to have)
        await sio.emit('manual', data={}, room=sid)


async def send_control(sid, steering_angle, throttle):
    await sio.emit('steer', data={
        'steering_angle': str(steering_angle),
        'throttle': str(throttle)
    }, room=sid)

if __name__ == '__main__':
    model = load_prediction_model(MODEL_PATH)

    if model:
        uvicorn.run(app, host=host, port=port)
    else:
        print("System exit: Could not load model.")
