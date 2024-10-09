import damask
import numpy as np
import scipy 


def average_over_grains(grid,values,method='mean'):
    """
    Computes the grain average for an NxM image.

    Args:
        grid (array): Image segmented by grains represented by integer values.
        values (array): Image with arbitrary values.
        method (str, optional): Average value to use - avaliable are 'mean','median' and 'mode'.

    Returns:
        average (array): Image with values averaged by grain.
    """

    # Create an empty map to store the average values
    average = np.zeros_like(grid,dtype=float)

    # Go grain by grain
    for i in np.unique(grid):

        # Find all the coordinates of that grain
        locations = np.argwhere(grid==i)

        # Find all the values of that grain
        values_grain = np.array([values[*location] for location in locations]) 

        # Find the average value
        if method=='mean':
            average_values_grain = np.mean(values_grain)
        elif method=='mode':
            average_values_grain = scipy.stats.mode(values_grain)[0]  
        elif method=='median':
            average_values_grain = np.median(values_grain)

        # Update the map with the new values
        for idx,location in enumerate(locations):

            average[*location] = average_values_grain 

    return average

def kernel_misorientation(grid,orientations):
    """
    Method to calculate the Kernel Average Misorientation (KAM) for an NxMx4 Image.

    Args:
        grid (array): Image segmented by grains represented by integer values.
        orientations (array): Image pixel values represented by quaternions.

    Returns:
        misorientation (array): Image where each pixel represents the KAM at that point.
    """

    # Pad the image maps to prevent periodic neighbours
    new_orientations = np.pad(orientations,pad_width=1,constant_values=np.nan)[:,:,1:-1]
    new_grid = np.pad(grid,pad_width=1,constant_values=-1)

    # Create a list to hold the misorientation values
    misorientation = []

    # We start from 1 and end on -1 to prevent catching the padded boundaries
    for i in range(1,new_orientations.shape[0]-1):
        for j in range(1,new_orientations.shape[1]-1):

            # Find the material value in the centre
            grid_centre = new_grid[i,j]

            # Find the orientation value in the centre
            centre = new_orientations[i,j]
            
            # Find the nearest neighbours and check they are valid
            neighbours =[new_orientations[i+1,j] if new_grid[i+1,j]==grid_centre else np.nan,
                        new_orientations[i-1,j] if new_grid[i-1,j]==grid_centre else np.nan,
                        new_orientations[i,j+1] if new_grid[i,j+1]==grid_centre else np.nan,
                        new_orientations[i,j-1] if new_grid[i,j-1]==grid_centre else np.nan]
            
            # Filter the neighbours
            neighbours = [neighbour for neighbour in neighbours if neighbour is not np.nan]
            N_neighbours = len(neighbours)

            # Using DAMASK, find the average disorientation
            if N_neighbours>0:

                centre =  damask.Orientation.from_quaternion(q=centre,lattice='cI')
                neighbours =  damask.Orientation.from_quaternion(q=np.array(neighbours),lattice='cI')

                misorientation_local = np.sum(centre.disorientation_angle(neighbours))/N_neighbours # Number of neighbours

                misorientation.append(misorientation_local*180/np.pi)
        
            if N_neighbours==0:

                misorientation.append(0)
            
            # Reshape the misorientation list back into an image
            KAM = np.array(misorientation).reshape(orientations.shape[0],orientations.shape[1])

    return KAM

def grain_reference_orientation_deviation(grid_surface,surface_orientation):
    """
    Method to calculate the Grain Reference Orientation Deviation (KAM) for an NxMx4 Image.

    Args:
        grid (array): Image segmented by grains represented by integer values.
        orientations (array): Image pixel values represented by quaternions.

    Returns:
        grod (array): Image where each pixel represents the GROD at that point.
    """

    # Create an empty map to store the disorientation angles
    grod = np.zeros_like(grid_surface,dtype=float)

    # Go grain by grain
    for i in np.unique(grid_surface):

        # Find all the coordinates of that grain
        locations = np.argwhere(grid_surface==i)

        # Find all the orientations of that grain
        surface_orientations_grain = np.array([surface_orientation[*location] for location in locations]) 

        # Convert to a DAMASK orientation object
        surface_orientations_grain = damask.Orientation.from_quaternion(q=surface_orientations_grain,lattice='cI')

        # Find the average orientation
        average_surface_orientations_grain = surface_orientations_grain.average()

        # Calculate a list of disorientation angles within that grain (in degrees)
        theta = average_surface_orientations_grain.disorientation_angle(surface_orientations_grain)*180.0/np.pi

        # Update the GROD map with the new values
        for idx,location in enumerate(locations):

            grod[*location] = theta[idx]

    return grod