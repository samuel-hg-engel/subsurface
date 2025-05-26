"""
Construction methods for the Voronoi class. 
"""
import numpy as np
import scipy
import math


#--------------------------------------------------
# Dervied property calculation
#--------------------------------------------------

def get_voronoi_surface(voronoi_matrix):

    shape = voronoi_matrix.shape

    if len(shape) == 2:
        return voronoi_matrix

    elif len(shape) == 3:
        return voronoi_matrix[:,:,0]
    
def get_grain_centre(voronoi_matrix,seed_locations,seed_ID,size,voxels):

    grain_centres = []

    seed_ID = np.linspace(0,len(seed_locations)-1,len(seed_locations)).astype(int)

    for ID in seed_ID:

        grain_centre = np.array(np.mean(np.where(voronoi_matrix==ID),axis=1))

        if math.isnan(np.sum(grain_centre)):
            grain_centres.append(seed_locations[ID])

        else:
            grain_centres.append(grain_centre*np.divide(size,voxels))

    return np.array(grain_centres)