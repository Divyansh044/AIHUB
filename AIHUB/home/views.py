import cv2
from django.shortcuts import render
from django.http import request, JsonResponse
import speech_recognition as sr
from tensorflow.keras.models import load_model # type: ignore
import numpy as np
from PIL import Image
import io
import tensorflow as tf
from keras.layers import LSTM # type: ignore
import os

import pytesseract
import transformers
import librosa
import torch
import IPython.display as display
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
import numpy as np
# Create your views here.
def home(request):
    return render (request,"index.html")


model=load_model('ml_models/Image_rec/img_classify.keras')
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']


tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

def preprocess_image(image):
    # Resize image to match model input dimensions
    image = image.resize((32, 32))
    # Convert image to numpy array and normalize pixel values
    image = np.array(image) / 255.0
    # Expand dimensions to match model input shape (add batch dimension)
    image = np.expand_dims(image, axis=0)
    return image



def predict_image(request):
    if request.method == 'POST':
        # Get uploaded image from request
        uploaded_image = request.FILES['image']
        # Open and preprocess the image
        image = Image.open(io.BytesIO(uploaded_image.read()))
        preprocessed_image = preprocess_image(image)
        # Make prediction
        predictions = model.predict(preprocessed_image)
        # Get predicted class index
        predicted_class_index = np.argmax(predictions)
        predicted_class=class_names[predicted_class_index]
        return JsonResponse({'predicted_class': predicted_class})

    return render(request, 'image_classification.html')

def speech_to_text(request):
    if request.method == 'POST':

        # Check if audio file is present in the request
        if 'audio_file' in request.FILES:
            audio_file = request.FILES['audio_file']

            try:
                input_values = tokenizer(audio_file, return_tensors = 'pt').input_values
                logits = model(input_values).logits
                predicted_ids = torch.argmax(logits, dim =-1)
                text=tokenizer.decode(predicted_ids[0])
                return JsonResponse({'text': text})
            except sr.UnknownValueError:
                return JsonResponse({'error': 'Unable to recognize speech'})
            except sr.RequestError as e:
                return JsonResponse({'error': f'Speech recognition service error: {e}'})
    else:
        return render(request,'speech_to_text.html')
    return JsonResponse({'error': 'Invalid request'}, status=400)

def sign_language(request):
   
    os.system("python ml_models/sign_lang/Sign-Language-Generation-From-Video-using-YOLOV5-master/yolov5-master/detect.py --weights ml_models/sign_lang/Sign-Language-Generation-From-Video-using-YOLOV5-master/yolov5-master/best.pt --img 416 --conf 0.4 --source 0")
    return render(request,'sign_lang.html')

def tts(request):
    return render(request,'text_to_speech.html')




















































































































































































































































































































































































































def handwritten(request):
    if request.method == 'POST' and request.FILES['image']:
        # Get the uploaded image from the request
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        uploaded_image = request.FILES['image']
        
        # Create a temporary file to store the uploaded image
        # temp_image_path = os.path.join(settings.MEDIA_ROOT, 'temp_image.png')
        # with open(temp_image_path, 'wb+') as temp_image:
        #     for chunk in uploaded_image.chunks():
        #         temp_image.write(chunk)
        
        # Open the temporary image using PIL (Python Imaging Library)
        img = Image.open(uploaded_image)
        
        # Use pytesseract to do OCR (Optical Character Recognition) on the image
        # It will extract text from the image
        Predicted_text = pytesseract.image_to_string(img)
        # Delete the temporary image file
        # os.remove(temp_image_path)
        return JsonResponse({'Predicted_text': Predicted_text})
    return render(request,'HWR.html')
    