# analyzer/forms.py
from django import forms

class UploadFileForm(forms.Form):
    file = forms.FileField(
        label='Upload Dataset (Satu per satu)', 
        required=True,
        widget=forms.FileInput(attrs={'class': 'form-control'})
    )
    
    sample_size = forms.IntegerField(
        label='Jumlah Sampel Data',
        min_value=100,
        max_value=50000,
        initial=5000,
        help_text="Jumlah baris yang akan diambil untuk analisis (agar cepat).",
        widget=forms.NumberInput(attrs={'class': 'form-control'})
    )
    
    n_estimators = forms.IntegerField(
        label='Jumlah Pohon (RF)', 
        min_value=10, 
        max_value=200, 
        initial=50,
        widget=forms.NumberInput(attrs={'class': 'form-control'})
    )