import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import scienceplots

plt.style.use('science')
plt.rc('text', usetex=False)
plt.rcParams['figure.dpi'] = 200

def plot_results(outputs,name,bounds=None,colorbar=False,cbar_ticks=None,cmap='coolwarm',interpolation='nearest'):
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

    fig,ax=plt.subplots(1,(length:=len(outputs)),sharex=True,sharey=True,figsize=(3*length,4))

    for i,matrix in enumerate(outputs):

        if length==1:
            ax=[ax]

        img = ax[i].imshow(matrix,origin="lower",cmap=cmap,vmin=minx,vmax=maxx,interpolation=interpolation)
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        
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