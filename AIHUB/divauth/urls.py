from django.urls import path
from divauth import views
from django.contrib.auth import views as auth_views
urlpatterns = [
    path("signup/",views.signup,name='signup'),
    path('login/', views.login,  name='login'),
    path('logout/', views.logout,name='logout'),
    path('activate/<uidb64>/<token>/',views.ActivateAccountView.as_view(),name='activate'),
    path('request-reset-email',views.RequestResetEmailView.as_view(),name='request-reset-email'),
    path('set-new-password/<uidb64>/<token>',views.SetNewPasswordView.as_view(),name='set-new-password')
]
