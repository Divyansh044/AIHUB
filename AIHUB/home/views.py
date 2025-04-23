import cv2
from django.shortcuts import render
from django.http import request, JsonResponse
import speech_recognition as sr
from tensorflow.keras.models import load_model # type: ignore
import numpy as np
from PIL import Image
import io
import os
import tensorflow as tf
from keras.layers import LSTM # type: ignore
import pytesseract
import transformers
import librosa
import torch
import IPython.display as display
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import decode_predictions, preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
from django.core.files.storage import FileSystemStorage
import os
from django.shortcuts import redirect
from django.views.decorators.csrf import csrf_exempt
from django.core.mail import send_mail
from django.contrib.auth.decorators import login_required

def home(request):
    return render (request,"index.html")

def contact_ajax_view(request):
    if request.method == 'POST':
        name = request.POST.get('name')
        email = request.POST.get('email')
        subject = request.POST.get('subject')
        message = request.POST.get('message')

        full_message = f"From: {name}\nEmail: {email}\n\n{message}"

        try:
            send_mail(
                subject,
                full_message,
                'email',  # FROM
                ['divyanshbagnwal@gmail.com'],  # TO
            )
            return JsonResponse({'success': True})
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)})
    return JsonResponse({'success': False, 'error': 'Invalid request'})
@login_required
def classify_image(request):
    if request.method == 'POST' and request.FILES['image']:
        img_file = request.FILES['image']
        fs = FileSystemStorage()
        file_path = fs.save(img_file.name, img_file)
        file_url = fs.url(file_path)

        # Load image for model
        img_path = fs.path(file_path)
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Load model
        model = tf.keras.applications.MobileNetV2(weights='imagenet')
        preds = model.predict(img_array)
        decoded = decode_predictions(preds, top=3)[0]

        result = [(label, float(prob)) for (_, label, prob) in decoded]
        os.remove(img_path)  # Clean up

        return render(request, 'image_classification.html', {'result': result, 'image': file_url})
    
    return render(request, 'image_classification.html')
# Create your views here.

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
model1 = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
@login_required
def speech_to_text(request):
    if request.method == 'POST':

        # Check if audio file is present in the request
        if 'audio_file' in request.FILES:
            audio_file = request.FILES['audio_file']

            try:
                # Read the content of the audio file
                audio_content,sampling_rate=librosa.load(audio_file,sr=16000)

                # Tokenize the audio content
                input_values = processor(audio_content, return_tensors="pt",sampling_rate=sampling_rate).input_values

            # Make prediction
                with torch.no_grad():  # Disable gradient calculation for efficiency
                   logits = model1(input_values).logits
                   predicted_ids = torch.argmax(logits, dim=-1)
                   text = processor.batch_decode(predicted_ids)[0]
                return JsonResponse({'text': text})
            except sr.UnknownValueError:
                return JsonResponse({'error': 'Unable to recognize speech'})
            except sr.RequestError as e:
                return JsonResponse({'error': f'Speech recognition service error: {e}'})
    else:
        return render(request, 'speech_to_text.html')
    return JsonResponse({'error': 'Invalid request'}, status=400)

@login_required
def sign_language(request):
   
    os.system("python ml_models/sign_lang/Sign-Language-Generation-From-Video-using-YOLOV5-master/yolov5-master/detect.py --weights ml_models/sign_lang/Sign-Language-Generation-From-Video-using-YOLOV5-master/yolov5-master/best.pt --img 416 --conf 0.4 --source 0")
    return redirect('home') 
@login_required
def tts(request):
    return render(request,'text_to_speech.html')

@login_required
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
    