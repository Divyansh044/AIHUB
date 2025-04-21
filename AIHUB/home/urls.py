from django.urls import path
from home import views
from .views import classify_image
from django.conf import settings
from django.conf.urls.static import static
urlpatterns = [
    path("",views.home,name="home"),
    path('speech-to-text/',views.speech_to_text,name='speech-to-text'),
    path('text-to-speech/',views.tts,name='text-to-speech'),
    path('handwritten_text/',views.handwritten,name='handwritten_text'),
    path('sign_language/',views.sign_language,name='sign_language'),
    path('classify/', classify_image, name='classify'),
]
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)