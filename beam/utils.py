# import matplotlib.pyplot as plt  
# import bas64 

# from  io import BytesIO

# import wradlib as wrl
# import matplotlib.pyplot as pl
# import matplotlib as mpl
# import warnings

# warnings.filterwarnings("ignore")
# try:
#     get_ipython().run_line_magic("matplotlib inline")
# except:
#     pl.ion()
# import numpy as np

# sitecoords = (-6.956542029485945, 32.85473170198925, 778)
# nrays = 360 # number of rays
# nbins = 1000 # number of range bins
# el = 0.3 # vertical antenna pointing angle (deg) # 0.523599
# bw = 1. # half power beam width (deg)           # 
# range_res = 100. # range resolution (meters)

# r = np.arange(nbins) * range_res
# beamradius = wrl.util.half_power_radius(r, bw)

# coord = wrl.georef.sweep_centroids(nrays, range_res, nbins, el)
# # print(coord.shape)
# # print(coord)

# coords = wrl.georef.spherical_to_proj(
#     coord[..., 0], coord[..., 1], coord[..., 2], sitecoords
# )
# polcoords = coords[..., :2]
# lon = coords[..., 0]
# lat = coords[..., 1]
# alt = coords[..., 2]
# rlimits = (lon.min(), lat.min(), lon.max(), lat.max())

# # rasterfile = wrl.util.get_wradlib_data_file('geo/bonn_gtopo.tif')
# rasterfile = "gtf.tif"

# ds = wrl.io.open_raster(rasterfile)
# rastervalues, rastercoords, proj = wrl.georef.extract_raster_dataset(
#     ds, nodata=-32768.0
# )
# # print(rastercoords)

# # Clip the region inside our bounding box
# ind = wrl.util.find_bbox_indices(rastercoords, rlimits)
# rastercoords = rastercoords[ind[1] : ind[3], ind[0] : ind[2], ...]
# rastervalues = rastervalues[ind[1] : ind[3], ind[0] : ind[2]]

# # Map rastervalues to polar grid points
# polarvalues = wrl.ipol.cart_to_irregular_spline(
#     rastercoords, rastervalues, polcoords, order=3, prefilter=False
# )

# PBB = wrl.qual.beam_block_frac(polarvalues, alt, beamradius)
# PBB = np.ma.masked_invalid(PBB)
# # print(PBB.shape)
# CBB = wrl.qual.cum_beam_block_frac(PBB)


# def get_graph() :
#     buffer  = BytesIO()
#     plt.savefig(buffer,format='png')
#     buffer.seek(0)
#     image_png = buffer.getvalue()
#     graph = base64.b64encode(image_png) 
#     graph = graph.decode('utf-8')

#     buffer.close()
#     return graph

# def get_plot(ax,x,y,z,e) :
#     ticks = (ax.get_xticks() / 1000).astype(int)
#     ax.set_xticklabels(ticks)
#     ticks = (ax.get_yticks() / 1000).astype(int)
#     ax.set_yticklabels(ticks)
#     ax.set_xlabel("Kilometers")
#     ax.set_ylabel("Kilometers")
#     if not cm is None:
#         pl.colorbar(cm, ax=ax)
#     if not title == "":
#         ax.set_title(title)
#     ax.grid()
    
    
    
#     graph = get_graph()
#     return graph

# def annotate_map(ax, cm=None, title=""):
#     ticks = (ax.get_xticks() / 1000).astype(int)
#     ax.set_xticklabels(ticks)
#     ticks = (ax.get_yticks() / 1000).astype(int)
#     ax.set_yticklabels(ticks)
#     ax.set_xlabel("Kilometers")
#     ax.set_ylabel("Kilometers")
#     if not cm is None:
#         pl.colorbar(cm, ax=ax)
#     if not title == "":
#         ax.set_title(title)
#     ax.grid()
    
    
    
    
'''




def home(request) :
    
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
    buffer.close()
    
    
   
    
    # Render the plot in a Django template
    # template = loader.get_template('beam/home.html')
    # context = {'plot_image': plot_image.getvalue()}
    # return HttpResponse(template.render(context, request))
    
    return render(request,'beam/home.html',{'plot_image': graph})



'''