from django.shortcuts import render
from .views import *

def mynewpage(request):
    return render(request,"index2.html")