from src.imports import *

def generate_coordinates(N_voxels):

    #return [list(index) for index in np.ndindex(*N_voxels)]

    if len(N_voxels)==3:
    
        coordinates = np.stack(np.meshgrid(np.linspace(0,N_voxels[0]-1,N_voxels[0]),
                                           np.linspace(0,N_voxels[1]-1,N_voxels[1]),
                                           np.linspace(0,N_voxels[2]-1,N_voxels[2]),
                                           indexing='ij'),
                                           axis=-1)

    elif len(N_voxels)==2:
    
        coordinates = np.stack(np.meshgrid(np.linspace(0,N_voxels[0]-1,N_voxels[0]),
                                           np.linspace(0,N_voxels[1]-1,N_voxels[1]),
                                           indexing='ij'),
                                           axis=-1)

    return coordinates

def generate_seeds(N_seeds,N_voxels):
    return N_voxels*np.random.random_sample(size=(N_seeds,len(N_voxels)))

def generate_voronoi(seeds,coordinates):
    
    # Create the seed tree
    seed_tree = scipy.spatial.KDTree(seeds)

    #voronoi = seed_tree.query(coordinates, workers = int(os.environ.get('OMP_NUM_THREADS',4)))[1]
    voronoi = seed_tree.query(coordinates, workers = 8 )[1]

    return voronoi    


def generate_weighted_voronoi(seeds,coordinates,weights,N_neighbours=10,weight_type='additive'):

    # Quick catch for unweighted voronoi
    if np.sum(weights)==0.0:
        return generate_voronoi(seeds,coordinates)

    else:
        # Create the KDTree
        seed_tree = scipy.spatial.KDTree(seeds)
    
        # Find the distance and colors
        distance,colors = seed_tree.query(coordinates, k=N_neighbours, workers = 8 )
        
        distance = distance.reshape(np.prod(distance.shape[0:-1]),distance.shape[-1])
        colors = colors.reshape(np.prod(colors.shape[0:-1]),colors.shape[-1])
    
        # Define the seed colors
        #seed_colors = np.linspace(0,len(seeds)-1,len(seeds))
        
        seed_colors = np.unique(colors)

        # Build a color weight dictionary - 312 us
        color_weights = dict(zip(seed_colors,weights)) 
    
        # Find the weights of each nearby seed - 2.12 s
        weight_matrix = np.vectorize(color_weights.get)(colors) # Need to speed up this computation

        #weight_matrix = np.vectorize(lambda k: color_weights[k])(colors)

        
        # Find the weighted distance to the nearest seed - Essentially free time
        if weight_type == 'multiplicative':
            weighted_distance = np.divide(distance,weight_matrix)

        elif weight_type == 'additive':
            weighted_distance = np.subtract(distance,weight_matrix)
        
        # Find the weighted colors
        weighted_colors = np.array([color[i] for color,i in zip(colors,np.argmin(weighted_distance,axis=1))])
    
        return weighted_colors.reshape(coordinates.shape[:-1])


def perturb_seeds(seeds,scale):
    return  np.array([np.random.normal(loc=seed,scale=scale) for seed in seeds])

def perturb_weights(weights,scale):
    return  np.array([np.abs(np.random.normal(loc=weight,scale=scale)) for weight in weights])

def calculate_accuracy(voronoi_reference,voronoi_perturbed,layer=-1):

    accuracy = (voronoi_reference[:,:,layer]==voronoi_perturbed[:,:,layer])*1
    
    return np.sum(accuracy)/np.multiply(*accuracy.shape)

def get_voronoi_surface(voronoi_matrix):

    shape = voronoi_matrix.shape

    if len(shape) == 2:
        return voronoi_matrix

    elif len(shape) == 3:
        return voronoi_matrix[:,:,-1]

def get_grain_centre(voronoi_matrix,seed_ID):

    grain_centre = np.array([np.mean(np.where(voronoi_matrix==ID),axis=1) for ID in seed_ID])

    return grain_centre

def numpyToVTK(data, output_file):
    data_type = vtk.VTK_INT
    shape = data.shape

    flat_data_array = data.flatten()
    vtk_data = numpy_support.numpy_to_vtk(num_array=flat_data_array, deep=True, array_type=data_type)

    img = vtk.vtkImageData()
    img.GetPointData().SetScalars(vtk_data)
    img.SetDimensions(shape[0], shape[1], shape[2])

    # Save the VTK file
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(output_file)
    writer.SetInputData(img)
    writer.Write()


