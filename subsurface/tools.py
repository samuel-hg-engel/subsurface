from subsurface.imports import *

def calculate_accuracy(reference_matrix,peturbed_matrix,layer=0):

    accuracy = (reference_matrix[:,:,layer]==peturbed_matrix[:,:,layer])*1

    accuracy = np.sum(accuracy)/np.multiply(*accuracy.shape)
    
    return accuracy


def calculate_displacement(objects,periodic=False):

    displacement = []

    for voronoi in objects:

        vector_displacement =  (voronoi.seed_locations - objects[0].seed_locations)

        if periodic==True:
            
            vector_displacement = np.mod(vector_displacement,voronoi.size/2)
        
        scalar_displacement = np.linalg.norm(vector_displacement,axis=1)
        
        displacement.append(scalar_displacement)

    displacement = np.array(displacement)

    return displacement


def extract_surfaces(matrix):

    return [matrix[0,:,:],matrix[-1,:,:],matrix[:,0,:],matrix[:,-1,:],matrix[:,:,0],matrix[:,:,-1]]


def find_grain_boundary(RVE,boundary_type='slim'):

    shape = RVE.shape

    boundary = np.zeros_like(RVE)

    for i in range(shape[0]-1):
        for j in range(shape[1]-1):

            centre = RVE[i,j]

            if boundary_type=='thick':
                neighbours = np.array([RVE[i+1,j],RVE[i-1,j],RVE[i,j+1],RVE[i,j-1]]) # Thick boundary
            
            elif boundary_type=='slim':
                neighbours = np.array([RVE[i+1,j],RVE[i,j+1]]) # Slim boundary

            if np.mean(neighbours) == centre:
                boundary[i,j] = 0

            else:
                boundary[i,j] = 1

    boundary = np.where(boundary==0, np.nan, boundary)

    return boundary

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
                # first order neighbours
                neighbours = np.array([
                                       #material[i+1,j,k],material[i-1,j,k],
                                       #material[i,j+1,k],material[i,j-1,k],
                                       #material[i,j,k+1],material[i,j,k-1],

                                        material[i-1,j,  k  ], # Close face
                                        material[i-1,j-1,k  ],
                                        material[i-1,j+1,k  ],
                                        material[i-1,j,  k+1],
                                        material[i-1,j-1,k+1],
                                        material[i-1,j+1,k+1],
                                        material[i-1,j,  k-1],
                                        material[i-1,j-1,k-1],
                                        material[i-1,j+1,k-1],

                                        #material[i  ,j,  k  ], # Centre face
                                        material[i  ,j-1,k  ],
                                        material[i  ,j+1,k  ],
                                        material[i  ,j,  k+1],
                                        material[i  ,j-1,k+1],
                                        material[i  ,j+1,k+1],
                                        material[i  ,j,  k-1],
                                        material[i  ,j-1,k-1],
                                        material[i  ,j+1,k-1],

                                        material[i+1,j,  k  ], # Far face
                                        material[i+1,j-1,k  ],
                                        material[i+1,j+1,k  ],
                                        material[i+1,j,  k+1],
                                        material[i+1,j-1,k+1],
                                        material[i+1,j+1,k+1],
                                        material[i+1,j,  k-1],
                                        material[i+1,j-1,k-1],
                                        material[i+1,j+1,k-1],
                                       ])

                neighbours = np.unique(neighbours).tolist()

                print(centre)
                neighbour_grains[centre].update(neighbours)

                if centre in neighbours:
                    neighbour_grains[centre].remove(centre) # Make sure to remove the centre as this is the self ID

    neighbour_grains = [list(grain) for grain in neighbour_grains]

    return neighbour_grains