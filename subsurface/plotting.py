import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import scienceplots

plt.style.use('science')
plt.rc('text', usetex=False)
plt.rcParams['figure.dpi'] = 200

def plot_results(outputs,axis_title=None,name=None,bounds=None,colorbar=False,cbar_ticks=None,cmap='coolwarm',interpolation='nearest',fig=None,ax=None):
    """
    Method to plot the results for a set of NxM images.

    Args:
        outputs (list): List of images.
        name (list): List of strings which correspond the the outputs.
        bounds (list, optional): List of [min,max] values of the image used for the colorbar.
        colorbar (bool, optional): Optional flag to include a colorbar in the plot.
        cbar_ticks (list, optional): Optional flag to specify specific colorbar ticks.
        cmap (str, optional): Colormap to use. Defaults to 'coolwarm'.

    Returns:
        fig,ax: matplotlib handles for further adjustment of the figure.
    """

    if bounds:
        minx=bounds[0]
        maxx=bounds[1]

    else:
        minx = np.min(outputs)
        maxx = np.max(outputs)

    if (fig and ax) is None:
        fig,ax=plt.subplots(1,(length:=len(outputs)),sharex=True,sharey=True,figsize=(3*length,4))
    else:
        pass
    
    plt.subplots_adjust(wspace=0.05, hspace=0)

    for i,matrix in enumerate(outputs):

        if length==1:
            ax=[ax]

        img = ax[i].imshow(matrix,origin="lower",cmap=cmap,vmin=minx,vmax=maxx,interpolation=interpolation)
        ax[i].set_xticks([])
        ax[i].set_yticks([])

        if not axis_title is None:
            ax[i].set_title(axis_title[i])
        
        if colorbar:
            divider = make_axes_locatable(ax[i])
            cax = divider.append_axes('bottom', size='5%', pad=0.05)

            if not cbar_ticks is None:
                cbar=fig.colorbar(img,cax=cax,ticks=cbar_ticks, orientation='horizontal',label=name)
                cbar.ax.minorticks_off()

            else:
                cbar=fig.colorbar(img,cax=cax, orientation='horizontal',label=name)

    fig.tight_layout()

    return fig,ax

import pyvista as pv

def initialise_cube():
    # Initialise the plot
    plotter = pv.Plotter(off_screen=True,notebook=False)

    # Initialise the grid
    grid = pv.ImageData()

    return plotter, grid


def plot_cube(matrix,colormap,plotter=None,grid=None,n_colors=None,light_intensity=[0.25,0.50,0.75]):

    # Initialise the plot if not done so already
    if (plotter==None) or (grid==None):

        # Initialise the plot
        plotter, grid = initialise_cube()

    # We must light our canvas with three lights for each visible face
    light = pv.Light(intensity=light_intensity[0]) # X Face
    light.set_direction_angle(0,0)
    plotter.add_light(light)

    light = pv.Light(intensity=light_intensity[1]) # Y Face 
    light.set_direction_angle(0,90)
    plotter.add_light(light)    

    light = pv.Light(intensity=light_intensity[2]) # Z Face
    light.set_direction_angle(90,0)
    plotter.add_light(light)

    # Load the matrix as values to the grid
    values=matrix

    grid.dimensions = np.array(values.shape) + 1

    grid.cell_data["values"] = values.flatten(order="F")  # Flatten the array

    # Rotate around y
    rotated = grid.rotate_y(180, inplace=False)

    # We must remember to set n_colors, or else we will only get 256 unique colors
    if not n_colors is None:
        plotter.add_mesh(rotated,cmap=colormap,n_colors=n_colors)

    else:
        plotter.add_mesh(rotated,cmap=colormap,n_colors=len(np.unique(values)))

    plotter.add_axes()
    plotter.remove_scalar_bar()

    image = plotter.show(return_img=True,interactive=False,auto_close=False)

    return image