from django import forms

class MyForm(forms.Form):
    name = forms.CharField(max_length=100)
    senario = forms.IntegerField()
    latitude  = forms.DecimalField(max_digits=40, decimal_places=30)
    longitude = forms.DecimalField(max_digits=40, decimal_places=30)
    altitude = forms.IntegerField()
    vertical_angle = forms.DecimalField(max_digits=8, decimal_places=5)
    
     
