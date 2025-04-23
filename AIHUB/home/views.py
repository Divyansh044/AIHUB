import cv2
from django.shortcuts import render
from django.http import JsonResponse
import speech_recognition as sr
from PIL import Image
import os
import speech_recognition as sr
import pytesseract
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from django.core.files.storage import FileSystemStorage
import os
from django.shortcuts import redirect
from django.views.decorators.csrf import csrf_exempt
from django.core.mail import send_mail
from django.contrib.auth.decorators import login_required
import requests
from decouple import config
IMAGGA_API_KEY = config("IMAGGA_API_KEY")
IMAGGA_API_SECRET = config("IMAGGA_API_SECRET")

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
    if request.method == 'POST' and request.FILES.get('image'):
        img_file = request.FILES['image']
        fs = FileSystemStorage()
        file_path = fs.save(img_file.name, img_file)
        img_path = fs.path(file_path)
        file_url = fs.url(file_path)

        with open(img_path, 'rb') as f:
            response = requests.post(
                'https://api.imagga.com/v2/tags',
                auth=(IMAGGA_API_KEY, IMAGGA_API_SECRET),
                files={'image': f}
            )

        os.remove(img_path)  # Clean up

        if response.status_code == 200:
            tags = response.json().get('result', {}).get('tags', [])[:3]
            result = [(tag['tag']['en'], tag['confidence']) for tag in tags]
            return render(request, 'image_classification.html', {'result': result, 'image': file_url})
        else:
            return JsonResponse({'error': 'Failed to classify image'}, status=500)

    return render(request, 'image_classification.html')
# Create your views here.

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
model1 = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
@csrf_exempt
@login_required
def speech_to_text(request):
   if request.method == 'POST':
        # Check if audio file is present in the request
        if 'audio_file' in request.FILES:
            audio_file = request.FILES['audio_file']

            recognizer = sr.Recognizer()
            try:
                with sr.AudioFile(audio_file) as source:
                    audio_data = recognizer.record(source)

                # Convert audio to text using Google's free API
                text = recognizer.recognize_google(audio_data)

                return JsonResponse({'text': text})

            except sr.UnknownValueError:
                return JsonResponse({'error': 'Unable to recognize speech'})
            except sr.RequestError as e:
                return JsonResponse({'error': f'Speech recognition service error: {e}'})
        else:
            return JsonResponse({'error': 'No audio file provided'}, status=400)
   else:
        return render(request, 'speech_to_text.html')

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
    