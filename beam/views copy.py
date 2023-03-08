from django.shortcuts import render

from . models import Location
# Create your views here.
import base64
import matplotlib.pyplot as plt
from django.http import HttpResponse
from django.template import loader
from io import BytesIO

def home(request) :
        # Create some data for the plot
    x = [1, 2, 3, 4, 5]
    y = [1, 4, 9, 16, 25]
    
    # Create the plot using Matplotlib
    fig, ax = plt.subplots()
    ax.plot(x, y)
    
    
    # Save the plot to a PNG image
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    png_image = buffer.getvalue()
    graph = base64.b64encode(png_image)
    graph = graph.decode('utf-8')
    buffer.close()
    
    # Render the plot in a Django template
    # template = loader.get_template('beam/home.html')
    # context = {'plot_image': plot_image.getvalue()}
    # return HttpResponse(template.render(context, request))
    
    return render(request,'beam/home.html',{'plot_image': graph})