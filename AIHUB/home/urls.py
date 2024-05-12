from django.urls import path
from home import views
urlpatterns = [
    path("",views.home,name="home"),
    path('speech-to-text/',views.speech_to_text,name='speech-to-text'),
    path('text-to-speech/',views.tts,name='text-to-speech'),
    path('image_rec/',views.predict_image,name='image_rec'),
    path('handwritten_text/',views.handwritten,name='handwritten_text'),
    path('sign_language/',views.sign_language,name='sign_language'),
]
