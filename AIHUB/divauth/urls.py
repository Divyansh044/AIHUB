from django.urls import path
from divauth import views
from django.contrib.auth import views as auth_views
urlpatterns = [
    path("signup/",views.signup,name='signup'),
    path('login/', auth_views.LoginView.as_view(template_name='auth/login.html', next_page='home'), name='login'),
    path('logout/', auth_views.LogoutView.as_view(next_page='home'), name='logout'),
    path('activate/<uidb64>/<token>/',views.ActivateAccountView.as_view(),name='activate'),
    path('request-reset-email',views.RequestResetEmailView.as_view(),name='request-reset-email'),
    path('set-new-password/<uidb64>/<token>',views.SetNewPasswordView.as_view(),name='set-new-password')
]
