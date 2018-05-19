from django import forms
from django.core import validators



class FormName(forms.Form):
    name = forms.CharField()
