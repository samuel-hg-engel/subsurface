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

def generate_voronoi(seeds,N_voxels,coordinates):
    
    # Create the seed tree
    seed_tree = scipy.spatial.KDTree(seeds)

    # Generate the voronoi from the seed_tree and coordinates
    #voronoi = seed_tree.query(coordinates, workers = int(os.environ.get('OMP_NUM_THREADS',4)))[1].reshape(N_voxels)
    
    voronoi = seed_tree.query(coordinates, workers = int(os.environ.get('OMP_NUM_THREADS',4)))[1]
    #voronoi = seed_tree.query(coordinates, workers = 5)[1].reshape(N_voxels)

    return voronoi    

def perturb_seeds(seeds,scale):
    return  np.array([np.random.normal(loc=seed,scale=scale) for seed in seeds])

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


