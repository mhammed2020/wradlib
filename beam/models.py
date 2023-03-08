from django.db import models

# Create your models here.


class Location(models.Model):
    
    name = models.CharField(max_length=100)
    senario = models.IntegerField(blank=True,null=True)
    latitude  = models.DecimalField(max_digits=30, decimal_places=6)
    longitude = models.DecimalField(max_digits=30, decimal_places=6)
    altitude = models.IntegerField()
    vertical_angle = models.DecimalField(max_digits=5, decimal_places=2)
    image_beam = models.ImageField(default = 'default.jpg' , upload_to ='senarios')
    image_couverture = models.ImageField(default = 'default.jpg' , upload_to ='senarios')


    def __str__(self):
        return " Radar de " + self.name + " : Senario " + str(self.senario)

