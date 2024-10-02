from subsurface.imports import *
from subsurface.tools import *
import random
import multiprocessing 

def Shuffle(material,orientations,iterations,exclude=None,return_full=False,minimize=False,family='cubic',lattice='cI'):
    """
    Function to shuffle the orientations in a representative volume element.

    Parameters
    ----------
    material : array (:,:,:)
        Three dimensional representative volume element as an array of material points.
    orientations : list
        List of quaternions that represent the grain orientations.
    iterations : int
        Number of iterations to shuffle for.
    exclude : list or None
        List of material indices to exlude from the shuffling operation.
    return_full : bool
        Request that the shuffle operation returns the misorientation distribution.
    minimize : bool
        Option to minimise the overall misorientation.
    family : str
        Slip family to consider - see damask.Orientation for full description.
    lattice : str
        Lattice to consider - see damask.Orientation for full description.

    Returns
    -------
    shuffled_orientations : list
        List of shuffled orientations.
    misorientations : list (optional)
        Unstructured list of misorientations.
    """

    # Quick catch to account for no exclusions
    exclude = [] if exclude is None else exclude

    # Firstly we need to identify what grains are near eachother.
    nearest_grains = find_neighbour_grains(material)

    for idx in range(iterations):

        # Determine the new orientations.
        orientations = find_new_arrangement(orientations,nearest_grains,exclude,minimize,family,lattice)

    if return_full:
        
        misorientation = find_misorientation(orientations,nearest_grains,family,lattice)

        return orientations,misorientation

    else: 
        return orientations


def find_misorientation(orientations,nearest_grains,family,lattice):
    """
    Wrapper for the find_local_misorientation code.

    Parameters
    ----------
    orientations : list
        List of quaternions that represent the grain orientations.
    neighbour_grains : list
        List of lists containing the nearest neighbour grains to each grain.
    family : str
        Slip family to consider - see damask.Orientation for full description.
    lattice : str
        Lattice to consider - see damask.Orientation for full description.

    Returns
    -------
    distribution : list
        Unstructured list of all misorientation angles.
    
    """

    distribution = []

    for idx in range(len(orientations)):
        
        distribution.extend(find_local_misorientation(orientations,nearest_grains,idx,family,lattice))

    return distribution





def find_local_misorientation(orientations,nearest_grains,seed_index,family,lattice):
    """
    For a given index, this function will calculate the local misorientation and return a list of angles.

    Parameters
    ----------
    orientations : list
        List of quaternions that represent the grain orientations.
    neighbour_grains : list
        List of lists containing the nearest neighbour grains to each grain.
    seed_index: int
        Index to calculate the local misorientation around.
    family : str
        Slip family to consider - see damask.Orientation for full description.
    lattice : str
        Lattice to consider - see damask.Orientation for full description.

    Returns
    -------
    omega : list
        Unstructured list of misorientation angles measured in degrees.
    
    """
    
    # Find all the equivalent orientations.
    reference_orientation =  damask.Orientation.from_quaternion(q=orientations[seed_index],lattice=lattice) # 230 ms
    
    # Find the neighbour orientations.
    nearest_orientations = [orientations[i] for i in nearest_grains]
    neighbour_orientations = damask.Orientation.from_quaternion(q=nearest_orientations[seed_index],lattice=lattice)

    # Find the misorientation angle
    #n,omega = reference_orientation.disorientation(neighbour_orientations).as_axis_angle(degrees=True,pair=True)

    omega = reference_orientation.disorientation_angle(neighbour_orientations)*180/np.pi
    
    return omega


def find_new_arrangement(orientations,nearest_grains,exclude,minimize,family,lattice):
    """
    Function to shuffle the orientations and calculate the misorientation.

    Parameters
    ----------
    orientations : list
        List of quaternions that represent the grain orientations.
    neighbour_grains : list
        List of lists containing the nearest neighbour grains to each grain.
    exclude : list or None
        List of material indices to exlude from the shuffling operation.
    minimize : bool
        Option to minimise the overall misorientation.
    family : str
        Slip family to consider - see damask.Orientation for full description.
    lattice : str
        Lattice to consider - see damask.Orientation for full description.

    Returns
    -------
    new_orientations : list
        List of shuffled orientations.
    """

    i,j = choose(0,len(orientations),exclude),choose(0,len(orientations),exclude)

    local_misorientation_home = find_local_misorientation(orientations,nearest_grains,seed_index=i,family=family,lattice=lattice) # What is the local misorientation on seed i?
    local_misorientation_away = find_local_misorientation(orientations,nearest_grains,seed_index=j,family=family,lattice=lattice) # What is the local misorientation on seed j?

    previous_misorientation = np.mean(local_misorientation_home)+np.mean(local_misorientation_away)

    new_orientations = switch(orientations,locations=[i,j])

    local_misorientation_home = find_local_misorientation(new_orientations,nearest_grains,seed_index=i,family=family,lattice=lattice)
    local_misorientation_away = find_local_misorientation(new_orientations,nearest_grains,seed_index=j,family=family,lattice=lattice)

    current_misorientation = np.mean(local_misorientation_home)+np.mean(local_misorientation_away)

    if minimize:
        if current_misorientation < previous_misorientation:
            return new_orientations

        else:
            return orientations
        
    else:
        return new_orientations

def find_neighbour_grains(material):
    """
    Function to find the neighbouring grains to grains in a material RVE.

    Parameters
    ----------
    material : array (:,:,:)
        Three dimensional representative volume element as an array of material points.

    Returns
    -------
    neighbour_grains : list
        List of lists containing the nearest neighbour grains to each grain.
    """

    shape = material.shape

    neighbour_grains = [set() for grain in np.unique(material)]

    # We start from 1 and end on -1 to prevent catching the boundaries
    for i in range(1,shape[0]-1):
        for j in range(1,shape[1]-1):
            for k in range(1,shape[2]-1):

                centre = material[i,j,k]
                
                # We only consider the first nearest neighbours
                neighbours = np.array([material[i+1,j,k],material[i-1,j,k],
                                       material[i,j+1,k],material[i,j-1,k],
                                       material[i,j,k+1],material[i,j,k-1],       
                                       ])

                neighbours = np.unique(neighbours).tolist()

                neighbour_grains[centre].update(neighbours)

                if centre in neighbours:
                    neighbour_grains[centre].remove(centre) # Make sure to remove the centre as this is the self ID

    neighbour_grains = [list(grain) for grain in neighbour_grains]

    return neighbour_grains

def choose(minimum,maximum,exclusions):
    """
    Function to sample from a flat distribution with exclusions.

    Parameters
    ----------
    minimum : int
        Minimum value of the distribution.
    maximum : int
        Maximum value of the distribution.
    exclusions : list
        List of values to exclude from the distribution.

    Returns
    -------
    choice : int
        Chosen value from the distribution.
    """

    choice = random.choice([i for i in range(minimum,maximum) if i not in exclusions])

    return choice

def switch(x,locations):
    """
    Function to switch rows in a list.

    Parameters
    ----------
    x : list
        List of values.
    locations : list of int of length (2)
        Two indices to swap.

    Returns
    -------
    x_new : list
        List of values with a switch.
    """
    
    i,j = locations
    
    x_new = copy.copy(x)

    x_new[[j,i]] = x_new[[i,j]] 
    
    return x_new

