from flask import Flask, render_template, Response, request, redirect, url_for, jsonify, session
from text_speech import text_to_speech
import cv2
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.models import model_from_json
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Set a secret key for session management
camera = cv2.VideoCapture(0)

# Mapping for prediction
d = {0: ' ', 1: 'Acting', 2: 'Bathing', 3: 'Catch',
     4: 'doctor', 5: 'elephant', 6: 'father', 7: 'goat',
     8: 'hat', 9: 'Ink', 10: 'Jack', 11: 'Kaviya',
     12: 'love', 13: 'quick', 18: 'rat', 19: 'say',
     20: 'Tittle', 21: 'Uncle', 22: 'Vicky', 23: 'Watch',
     24: 'Xerox', 25: 'Yak', 26: 'Zebra'}

# Coordinates for the gesture recognition window
upper_left = (335, 3)
bottom_right = (635, 303)

# Load the pre-trained model
with open('model-bw.json', 'r') as json_file:
    loaded_model_json = json_file.read()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model-bw.h5")

# Function to process the image for gesture recognition
def function(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 1)
    th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    ret, res = cv2.threshold(th3, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return res

# Function to generate frames for video stream
def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            frame = cv2.flip(frame, 1)
            r = cv2.rectangle(frame, upper_left, bottom_right, (0, 0, 0), 5)
            rect_img = frame[upper_left[1]:bottom_right[1], upper_left[0]:bottom_right[0]]
            sketcher_rect = rect_img
            sketcher_rect = function(sketcher_rect)
            sketcher_rect_rgb = cv2.cvtColor(sketcher_rect, cv2.COLOR_GRAY2RGB)
            frame[upper_left[1]:bottom_right[1], upper_left[0]:bottom_right[0]] = sketcher_rect_rgb

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Initialize variables
l = []
str1 = ""

# Route to set voice gender
@app.route('/set_voice', methods=['POST'])
def set_voice():
    session['voice_gender'] = request.form['voice']
    return redirect(url_for('index'))

# Route to set language
@app.route('/set_language', methods=['POST'])
def set_language():
    session['language'] = request.form['language']
    return redirect(url_for('index'))

# Route for prediction
@app.route('/predict', methods=['POST'])
def predictions():
    global l, str1  # Make sure we are using the global variables
    success, frame = camera.read()
    if not success:
        return jsonify({'error': 'Failed to capture image from camera'}), 500

    frame = cv2.flip(frame, 1)
    r = cv2.rectangle(frame, upper_left, bottom_right, (0, 0, 0), 5)
    rect_img = frame[upper_left[1]:bottom_right[1], upper_left[0]:bottom_right[0]]
    sketcher_rect = rect_img
    sketcher_rect = function(sketcher_rect)
    sketcher_rect_rgb = cv2.cvtColor(sketcher_rect, cv2.COLOR_GRAY2RGB)
    frame[upper_left[1]:bottom_right[1], upper_left[0]:bottom_right[0]] = sketcher_rect_rgb

    sketcher_rect = cv2.resize(sketcher_rect, (128, 128))
    x = image.img_to_array(sketcher_rect)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0
    pre = loaded_model.predict(x)
    p_test = np.argmax(pre)
    a = d[p_test]
    l.append(a)
    str1 = "".join(l)

    return jsonify({
        'success': True,
        'pred': str1,
    })

# Route to stop prediction and perform text-to-speech
@app.route('/stop', methods=['POST'])
def stopping():
    global l, str1  # Declare l and str1 as global to modify them within this function
    voice_gender = session.get('voice_gender', 'Female')
    language = session.get('language', 'en')

    # Combine the detected gestures into a word
    str1 = "".join(l)  # Update str1 with the combined string from list l

    # Perform text-to-speech on the final word
    if str1.strip():  # Only proceed if there's a word to announce
        text_to_speech(str1, voice_gender, language)

    # Clear the prediction list and the final word string
    l.clear()  # Clear the list to reset predictions
    str1 = ""  # Clear the string to reset the displayed word

    # Return a JSON response with the cleared prediction
    return jsonify({
        'success': True,
        'message': 'Stopped prediction and performed text-to-speech.',
        'pred': ""  # Clear the result display
    })


# Route for homepage
@app.route('/')
def home():
    return render_template('home.html')

# Route for the main translation page
@app.route('/translate')
def index():
    voice_gender = session.get('voice_gender', 'Female')
    language = session.get('language', 'en')
    return render_template('index.html', voice_gender=voice_gender, language=language)

@app.route('/play_greeting', methods=['POST'])
def play_greeting():
    data = request.get_json()
    voice_gender = data.get('voice_gender', 'Female')
    language = data.get('language', 'en')
    text_to_speech('Hi there please show the hand gesture in the provided space', voice_gender, language)
    return jsonify({'success': True})

# Route to stream video feed
@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
