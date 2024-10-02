from subsurface.imports import *
from subsurface.tools import *
import random

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

    distribution = []

    for idx in range(len(orientations)):
        
        distribution.extend(find_local_misorientation(orientations,nearest_grains,idx,family,lattice))

    return distribution


def find_local_misorientation(orientations,nearest_grains,seed_index,family,lattice):

    i = seed_index

    nearest_orientations = [orientations[i] for i in nearest_grains]
    
    # Find all the equivalent orientations.
    reference_orientation =  np.array(damask.Orientation.from_quaternion(q=orientations[i],family=family,lattice=lattice).equivalent) # 230 ms
    
    # Find the neighbour orientations.
    neighbour_orientations = nearest_orientations[i]
    
    # Take the dot product.
    layer_matrix = np.array([np.outer(reference_orientation[:,j],neighbour_orientations[:,j]) for j in range(4)])

    # Use the dot product to calculate the angle between the quaternions, but only keep the smallest angle of the equivalent quaternions.
    distribution = np.min((2*np.arccos(np.abs(np.sum(layer_matrix,axis=0)))*180/np.pi),axis=0)

    return distribution


def find_new_arrangement(orientations,nearest_grains,exclude,minimize,family,lattice):

    i,j = choose(len(orientations),exclude),choose(len(orientations),exclude)

    local_misorientation_home = find_local_misorientation(orientations,nearest_grains,seed_index=i,family=family,lattice=lattice) # What is the local misorientation on seed i?
    local_misorientation_away = find_local_misorientation(orientations,nearest_grains,seed_index=j,family=family,lattice=lattice) # What is the local misorientation on seed j?

    previous_misorientation = np.mean(local_misorientation_home)+np.mean(local_misorientation_away)

    new_orientations = shuffle(orientations,locations=[i,j])

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

def find_neighbour_grains(RVE):

    shape = RVE.shape

    neighbour_grains = [set() for grain in np.unique(RVE)]

    # We start from 1 and end on -1 to prevent catching the boundaries
    for i in range(1,shape[0]-1):
        for j in range(1,shape[1]-1):
            for k in range(1,shape[2]-1):

                centre = RVE[i,j,k]
                
                neighbours = np.array([RVE[i+1,j,k],RVE[i-1,j,k],
                                       RVE[i,j+1,k],RVE[i,j-1,k],
                                       RVE[i,j,k+1],RVE[i,j,k-1],       
                                       ])

                neighbours = np.unique(neighbours).tolist()

                neighbour_grains[centre].update(neighbours)

                if centre in neighbours:
                    neighbour_grains[centre].remove(centre) # Make sure to remove the centre as this is the self ID

    neighbour_grains = [list(grain) for grain in neighbour_grains]

    return neighbour_grains

def choose(max_value,exclusions):

    return random.choice([i for i in range(0,max_value) if i not in exclusions])

def shuffle(x,return_index=False,locations=False):

    x_new = copy.copy(x)

    # pick two rows to swap
    if locations:
        row1,row2=locations
    
    else:
        row1,row2 = np.random.randint(low=0,high=len(x),size=2)

    x_new[[row2,row1]] = x_new[[row1,row2]] 

    if return_index:
        return x_new,[row1,row2]
    else:
        return x_new

