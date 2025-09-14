# python -m flask run --no-reload

from flask import Flask, render_template, request, redirect, url_for, Response, jsonify,Markup
import cv2
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
import os
from utils1.disease import disease_dic
# from utils1.fertilizer import fertilizer_dic
import requests
import config
import pickle
import io
import torch
from torchvision import transforms
from PIL import Image
from utils1.model import ResNet9


import serial
import time
arduino = serial.Serial(port='COM3', baudrate=115200, timeout=.1)

# app = Flask(_name_)
# if _name_ == '_main_':
#     app.run(debug=False) 

disease_classes = ['Apple___Apple_scab',
                   'Apple___Black_rot',
                   'Apple___Cedar_apple_rust',
                   'Apple___healthy',
                   'Blueberry___healthy',
                   'Cherry_(including_sour)_Powdery_mildew',
                   'Cherry_(including_sour)_healthy',
                   'Corn_(maize)_Cercospora_leaf_spot Gray_leaf_spot',
                   'Corn_(maize)Common_rust',
                   'Corn_(maize)_Northern_Leaf_Blight',
                   'Corn_(maize)_healthy',
                   'Grape___Black_rot',
                   'Grape__Esca(Black_Measles)',
                   'Grape__Leaf_blight(Isariopsis_Leaf_Spot)',
                   'Grape___healthy',
                   'Orange__Haunglongbing(Citrus_greening)',
                   'Peach___Bacterial_spot',
                   'Peach___healthy',
                   'Pepper,bell__Bacterial_spot',
                   'Pepper,bell__healthy',
                   'Potato___Early_blight',
                   'Potato___Late_blight',
                   'Potato___healthy',
                   'Raspberry___healthy',
                   'Soybean___healthy',
                   'Squash___Powdery_mildew',
                   'Strawberry___Leaf_scorch',
                   'Strawberry___healthy',
                   'Tomato___Bacterial_spot',
                   'Tomato___Early_blight',
                   'Tomato___Late_blight',
                   'Tomato___Leaf_Mold',
                   'Tomato___Septoria_leaf_spot',
                   'Tomato___Spider_mites Two-spotted_spider_mite',
                   'Tomato___Target_Spot',
                   'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                   'Tomato___Tomato_mosaic_virus',
                   'Tomato___healthy']

disease_model_path = 'models/plant_disease_model.pth'
disease_model = ResNet9(3, len(disease_classes))
disease_model.load_state_dict(torch.load(
disease_model_path, map_location=torch.device('cpu')))
disease_model.eval()


def predict_image(img, model=disease_model):
    """
    Transforms image to tensor and predicts disease label
    :params: image
    :return: prediction (string)
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])
    image = Image.open(io.BytesIO(img))
    img_t = transform(image)
    img_u = torch.unsqueeze(img_t, 0)

    # Get predictions from model
    yb = model(img_u)
    # Pick index with highest probability
    _, preds = torch.max(yb, dim=1)
    prediction = disease_classes[preds[0].item()]
    # Retrieve the class label
    return prediction

app = Flask(_name_)

UPLOAD_FOLDER = 'static/images/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
leaf_count = 0  # Global variable to store leaf count
camera = cv2.VideoCapture(0)

def process_frame(frame):
    global leaf_count
    # Convert to gray-scale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian Blur
    blur = cv2.GaussianBlur(gray, (23, 23), 0)
    # Apply Thresholding
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Find the background area
    kernel = np.ones((2, 2), np.uint8)
    erode = cv2.erode(thresh, kernel, iterations=2)
    bg = cv2.dilate(erode, kernel, iterations=4)

    # Find the foreground area
    dist_transform = cv2.distanceTransform(thresh, cv2.DIST_L2, 3)
    fg = cv2.threshold(dist_transform, 0.15 * dist_transform.max(), 255, 0)[1]
    fg = np.uint8(fg)

    # Label and isolate the leaves using mask size
    label_objects = ndi.label(fg)[0]
    sizes = np.bincount(label_objects.ravel())
    mask_sizes = sizes > 1000
    mask_sizes[0] = 0
    fg1 = mask_sizes[label_objects]
    fg1 = np.uint8(fg1)

    # Draw contours on leaves and count them
    contours, _ = cv2.findContours(fg1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(frame, contours, -1, (0, 0, 255), 1)
    
    # Update the global leaf count
    leaf_count = len(contours)
    print(f"Updated Leaf Count: {leaf_count}")  # Debug statement
    
    return frame



def generate_frames():
    while True:
        success, frame = camera.read()  # Capture frame-by-frame
        if not success:
            break
        else:
            # Process the frame
            processed_frame = process_frame(frame)
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            frame = buffer.tobytes()

            # Yield the image as a response
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template ("index.html")

@app.route('/pg')
def growth():
    return render_template('index2.html')

@app.route('/choose', methods=['POST'])
def choose_action():
    choice = request.form.get('action_choice')
    if choice == 'camera':
        return redirect(url_for('camera_page'))
    elif choice == 'upload':
        return redirect(url_for('upload_page'))
    return redirect(url_for('index'))

@app.route('/camera')
def camera_page():
    return render_template('camera.html')   

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/upload', methods=['GET', 'POST'])
def upload_page():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            # Process the uploaded image
            image = cv2.imread(filepath)
            processed_image = process_frame(image)
            processed_filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_' + file.filename)
            cv2.imwrite(processed_filepath, processed_image)

            # Format the leaf count and age as two-digit values
            leaf_count_str = f"{leaf_count:02d}"
            age = leaf_count_str  # Use leaf count directly as the age

            return render_template('result.html', uploaded_image=processed_filepath, leaf_count=leaf_count_str, age=age)
    return render_template('upload.html')


@app.route('/leaf_count')
def get_leaf_count():
    global leaf_count
    return jsonify({'leaf_count': leaf_count})

#disease prediction 

# @ app.route('/dp',methods=["GET","POST"])
# def home():
#     title = "Disease Prediction"
#     if request.method == 'POST':
#         if 'file' not in request.files:
#             return redirect(request.url)
#         file = request.files.get('file')
#         if not file:
#             return render_template('disease.html', title=title)
#         try:
#             img = file.read()

#             prediction = predict_image(img)

#             prediction = Markup(str(disease_dic[prediction]))
#             print("datasent")
#             arduino.write(b'A')
#             arduino.write(b'B')
#             time.sleep(3)
        
#             return render_template('disease-result.html', prediction=prediction, title=title)
#         except:
#             pass
#     return render_template('disease.html', title=title)

@app.route('/dp', methods=['GET', 'POST'])
def disease_prediction():
    title = 'Harvestify - Disease Detection'

    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files.get('file')
        if not file:
            return render_template('disease.html', title=title)
        
        try:
            img = file.read()
            prediction = predict_image(img)

            if "healthy" not in prediction:
                prediction_info = Markup(str(disease_dic[prediction]))
                # arduino.write(b'a')
                # time.sleep(2)
                print("datasent")
                arduino.write(b'A')
                arduino. write(b'B')
                time.sleep(3)
            else:
                prediction_info = "No disease detected."

            return render_template('disease-result.html', prediction=prediction_info, title=title)
        
        except Exception as e:
            print(f"Error: {e}")
            return render_template('disease.html', title=title)

    return render_template('disease.html', title=title)

if _name=="main_":
    app.run(debug=True,port=4000)