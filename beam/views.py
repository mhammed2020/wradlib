from django.shortcuts import render

from . models import Location
# Create your views here.
import base64
# import matplotlib.pyplot as plt
from django.http import HttpResponse
from django.template import loader
from io import BytesIO

import wradlib as wrl
import matplotlib.pyplot as pl
import matplotlib as mpl
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from PIL import Image
import osgeo.gdal as gdal
# try:
#     get_ipython().run_line_magic("matplotlib inline")
# except:
#     pl.ion()
    
# import matplotlib.pyplot as plt
# plt.switch_backend('agg')

pl.switch_backend('agg')

sitecoords = (-7.576858296279283, 33.36866931346493, 230)
# -6.956881892616707, 32.855358085580896 
# sitecoords = (-6.956542029485945, 32.85473170198925, 778)
nrays = 360 # number of rays
nbins = 1000 # number of range bins
el = 0.7 # vertical antenna pointing angle (deg) # 0.523599
bw = 1. # half power beam width (deg)           # 
range_res = 100. # range resolution (meters)


from . forms import MyForm


def my_view(request):
    # form = MyForm()
  
    if request.method == 'POST':
        # print("**************",request.POST)
        form = MyForm(request.POST)
        if form.is_valid():
            # name =  request.POST.get('name')
            #senario =  request.POST.get('senario')
            latitude =  float(request.POST.get('latitude'))
            longitude =  float(request.POST.get('longitude'))
            altitude =  float(request.POST.get('altitude'))
            vertical_angle = float(request.POST.get('vertical_angle'))
            
            #Location.objects.create(name=name,senario=senario,latitude=latitude,longitude=longitude,altitude=altitude,vertical_angle=vertical_angle)
            
            sitecoords = (longitude, latitude, altitude)
            # -6.956881892616707, 32.855358085580896 
            # sitecoords = (-6.956542029485945, 32.85473170198925, 778)
            nrays = 360 # number of rays
            nbins = 1000 # number of range bins
            el = vertical_angle # vertical antenna pointing angle (deg) # 0.523599
            bw = 1. # half power beam width (deg)           # 
            range_res = 100. # range resolution (meters)

    
            # Implement my view her 
            
            r = np.arange(nbins) * range_res
            beamradius = wrl.util.half_power_radius(r, bw)
            coord = wrl.georef.sweep_centroids(nrays, range_res, nbins, el)
            coords = wrl.georef.spherical_to_proj(
            coord[..., 0], coord[..., 1], coord[..., 2], sitecoords
            )

            polcoords = coords[..., :2]
            lon = coords[..., 0]
            lat = coords[..., 1]
            alt = coords[..., 2]
            rlimits = (lon.min(), lat.min(), lon.max(), lat.max())
            # rasterfile = "gtf.tif"
            # rasterfile = Image.open("C:\\Users\\jeddou\\ssoa\\src\\beam\\gtf.tif") 
            
            rasterfile = "C:\\Users\\jeddou\\ssoa\\src\\beam\\gtf.tif"
            dataset = gdal.OpenEx(rasterfile, gdal.OF_RASTER)
            
            ds = wrl.io.open_raster(rasterfile)
            rastervalues, rastercoords, proj = wrl.georef.extract_raster_dataset(
                ds, nodata=-32768.0
            )
            # print(rastercoords)

            # Clip the region inside our bounding box
            ind = wrl.util.find_bbox_indices(rastercoords, rlimits)
            rastercoords = rastercoords[ind[1] : ind[3], ind[0] : ind[2], ...]
            rastervalues = rastervalues[ind[1] : ind[3], ind[0] : ind[2]]

            # Map rastervalues to polar grid points
            polarvalues = wrl.ipol.cart_to_irregular_spline(
                rastercoords, rastervalues, polcoords, order=3, prefilter=False
            )

            PBB = wrl.qual.beam_block_frac(polarvalues, alt, beamradius)
            PBB = np.ma.masked_invalid(PBB)
            CBB = wrl.qual.cum_beam_block_frac(PBB)

            # just a little helper function to style x and y axes of our maps
            def annotate_map(ax, cm=None, title=""):
                ticks = (ax.get_xticks() / 1000).astype(int)
                ax.set_xticklabels(ticks)
                ticks = (ax.get_yticks() / 1000).astype(int)
                ax.set_yticklabels(ticks)
                ax.set_xlabel("Kilometers")
                ax.set_ylabel("Kilometers")
                if not cm is None:
                    pl.colorbar(cm, ax=ax)
                if not title == "":
                    ax.set_title(title)
                ax.grid()
                
                
            fig = pl.figure(figsize=(15, 12))

            # create subplots
            ax1 = pl.subplot2grid((2, 2), (0, 0))
            ax2 = pl.subplot2grid((2, 2), (0, 1))
            ax3 = pl.subplot2grid((2, 2), (1, 0), colspan=2, rowspan=1)

            # azimuth angle
            angle = 140

            # Plot terrain (on ax1)
            ax1, dem = wrl.vis.plot_ppi(
                polarvalues, ax=ax1, r=r, az=coord[:, 0, 1], cmap=mpl.cm.terrain, vmin=0.0
            )
            ax1.plot(
                [0, np.sin(np.radians(angle)) * 1e5], [0, np.cos(np.radians(angle)) * 1e5], "r-"
            )
            ax1.plot(sitecoords[0], sitecoords[1], "ro")
            annotate_map(ax1, dem, "Terrain within {0} km range".format(np.max(r / 1000.0) + 0.1))

            # Plot CBB (on ax2)
            ax2, cbb = wrl.vis.plot_ppi(
                CBB, ax=ax2, r=r, az=coord[:, 0, 1], cmap=mpl.cm.PuRd, vmin=0, vmax=1
            )
            annotate_map(ax2, cbb, "Beam-Blockage Fraction")

            # Plot single ray terrain profile on ax3
            (bc,) = ax3.plot(r / 1000.0, alt[angle, :], "-b", linewidth=3, label="Beam Center")
            (b3db,) = ax3.plot(
                r / 1000.0,
                (alt[angle, :] + beamradius),
                ":b",
                linewidth=1.5,
                label="3 dB Beam width",
            )
            ax3.plot(r / 1000.0, (alt[angle, :] - beamradius), ":b")
            ax3.fill_between(r / 1000.0, 0.0, polarvalues[angle, :], color="0.75")
            ax3.set_xlim(0.0, np.max(r / 1000.0) + 0.1)
            ax3.set_ylim(0.0, 3000)
            ax3.set_xlabel("Range (km)")
            ax3.set_ylabel("Altitude (m)")
            ax3.grid()

            axb = ax3.twinx()
            (bbf,) = axb.plot(r / 1000.0, CBB[angle, :], "-k", label="BBF")
            axb.set_ylabel("Beam-blockage fraction")
            axb.set_ylim(0.0, 1.0)
            axb.set_xlim(0.0, np.max(r / 1000.0) + 0.1)


            legend = ax3.legend(
                (bc, b3db, bbf),
                ("Beam Center", "3 dB Beam width", "BBF"),
                loc="upper left",
                fontsize=10,
            )
            
            
            
            
              # Save the plot to a PNG image
            buffer = BytesIO()
            pl.savefig(buffer, format='png')
            buffer.seek(0)
            png_image = buffer.getvalue()
            graph = base64.b64encode(png_image)
            graph = graph.decode('utf-8')
            # buffer.close()


            # return redirect('my_other_view')
            # return render(request, 'beam/index.html',{'plot_image': graph} ) # 
            
            
            # Couverture
            def height_formatter(x, pos):
                x = (x - 6370000) / 1000
                fmt_str = "{:g}".format(x)
                return fmt_str


            def range_formatter(x, pos):
                x = x / 1000.0
                fmt_str = "{:g}".format(x)
                return fmt_str
            
            fig = pl.figure(figsize=(10, 6))

            cgax, caax, paax = wrl.vis.create_cg(fig=fig, rot=0, scale=1)

            # azimuth angle
            angle = 225

            # fix grid_helper
            er = 6370000
            gh = cgax.get_grid_helper()
            gh.grid_finder.grid_locator2._nbins = 80
            gh.grid_finder.grid_locator2._steps = [1, 2, 4, 5, 10]

            # calculate beam_height and arc_distance for ke=1
            # means line of sight
            bhe = wrl.georef.bin_altitude(r, 0, sitecoords[2], re=er, ke=1.0)
            ade = wrl.georef.bin_distance(r, 0, sitecoords[2], re=er, ke=1.0)
            nn0 = np.zeros_like(r)
            # for nice plotting we assume earth_radius = 6370000 m
            ecp = nn0 + er
            # theta (arc_distance sector angle)
            thetap = -np.degrees(ade / er) + 90.0

            # zero degree elevation with standard refraction
            bh0 = wrl.georef.bin_altitude(r, 0, sitecoords[2], re=er)

            # plot (ecp is earth surface normal null)
            (bes,) = paax.plot(thetap, ecp, "-k", linewidth=3, label="Earth Surface NN")
            (bc,) = paax.plot(thetap, ecp + alt[angle, :], "-b", linewidth=3, label="Beam Center")
            (bc0r,) = paax.plot(thetap, ecp + bh0 + alt[angle, 0], "-g", label="0 deg Refraction")
            (bc0n,) = paax.plot(
                thetap, ecp + bhe + alt[angle, 0], "-r", label="0 deg line of sight"
            )
            (b3db,) = paax.plot(
                thetap, ecp + alt[angle, :] + beamradius, ":b", label="+3 dB Beam width"
            )
            paax.plot(thetap, ecp + alt[angle, :] - beamradius, ":b", label="-3 dB Beam width")

            # orography
            paax.fill_between(thetap, ecp, ecp + polarvalues[angle, :], color="0.75")

            # shape axes
            cgax.set_xlim(0, np.max(ade))
            cgax.set_ylim([ecp.min() - 1000, ecp.max() + 2500])
            caax.grid(True, axis="x")
            cgax.grid(True, axis="y")
            cgax.axis["top"].toggle(all=False)
            caax.yaxis.set_major_locator(
                mpl.ticker.MaxNLocator(steps=[1, 2, 4, 5, 10], nbins=20, prune="both")
            )
            caax.xaxis.set_major_locator(mpl.ticker.MaxNLocator())
            caax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(height_formatter))
            caax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(range_formatter))

            caax.set_xlabel("Range (km)")
            caax.set_ylabel("Altitude (km)")

            legend = paax.legend(
                (bes, bc0n, bc0r, bc, b3db),
                (
                    "Earth Surface NN",
                    "0 deg line of sight",
                    "0 deg std refraction",
                    "Beam Center",
                    "3 dB Beam width",
                ),
                loc="upper left",
                fontsize=10,
            )
            
            
            
          

            # Save the plot to a PNG image
            buffer2 = BytesIO()
            pl.savefig(buffer2, format='png')
            buffer2.seek(0)
            png_image2 = buffer2.getvalue()
            graph2 = base64.b64encode(png_image2)
            graph2 = graph2.decode('utf-8')
            # buffer2.close()
            
          

            # create a new instance of your model and save it with the Matplotlib plot as its data field
             # Create new plot object model instance
            from django.core.files.uploadedfile import InMemoryUploadedFile 
            image_file = InMemoryUploadedFile(buffer, None, 'plot.png', 'image/png', buffer.getbuffer().nbytes, None)
            image_file2 = InMemoryUploadedFile(buffer2, None, 'plot.png', 'image/png', buffer2.getbuffer().nbytes, None)

            name =  request.POST.get('name')
            senario =  request.POST.get('senario')
            latitude =  request.POST.get('latitude')
            longitude =  request.POST.get('longitude')
            altitude =  request.POST.get('altitude')
            vertical_angle = request.POST.get('vertical_angle')
            
            obj = Location(  name = name  ,
                senario = senario,
                latitude  = latitude ,
                longitude = longitude, 
                altitude = altitude ,
                vertical_angle = vertical_angle,
                image_beam = image_file,
                image_couverture = image_file2 )
            obj.save()

        

            return render(request, 'beam/index.html',{'plot_image': graph,'plot_image2': graph2} ) # 
    
    else :
        return render(request, 'beam/index.html',{'form': MyForm()} ) # 
            
        
    # else :
    #     form = MyForm()
        # return render(request, 'beam/index.html',{'form': MyForm(),'plot_image': graph} ) # 
          
            # Do something with the values



def my_view_helper(request):

    return render(request, 'beam/index.html',{'form': MyForm()} ) # 'plot_image': graph


def list_location(request):
    
    return render(request, 'beam/list_location.html',{'locations': Location.objects.all()} ) # 'plot_image': graph
