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