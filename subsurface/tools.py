from subsurface.imports import *

def generate_coordinates(N_voxels,size,origin=np.array([0,0,0])):

    
    if len(N_voxels)==3:
    
        start = origin                  + size/N_voxels*0.5 # Add voxel centre offset
        end = origin + np.array(size)   - size/N_voxels*0.5 # Add voxel offset
    
        coordinates = np.stack(np.meshgrid(np.linspace(start[0],end[0],N_voxels[0]),
                                           np.linspace(start[1],end[1],N_voxels[1]),
                                           np.linspace(start[2],end[2],N_voxels[2]),
                                           indexing='ij'),
                                           axis=-1)

    elif len(N_voxels)==2:

        origin = np.array([0,0])
                                           
        start = origin                  + size/N_voxels*0.5 # Add voxel centre offset
        end = origin + np.array(size)   - size/N_voxels*0.5 # Add voxel offset

        coordinates = np.stack(np.meshgrid(np.linspace(start[0],end[0],N_voxels[0]),
                                           np.linspace(start[1],end[1],N_voxels[1]),
                                           indexing='ij'),
                                           axis=-1)

    return coordinates

def generate_seeds(seeds,size):

    raw_seeds = np.random.random_sample(size=(seeds,len(size)))

    return np.array(raw_seeds*size)

def generate_voronoi(seeds,coordinates):
    
    # Create the seed tree
    seed_tree = scipy.spatial.KDTree(seeds)

    #voronoi = seed_tree.query(coordinates, workers = int(os.environ.get('OMP_NUM_THREADS',4)))[1]
    voronoi = seed_tree.query(coordinates.reshape(-1,coordinates.shape[-1]), workers = 8 )[1]

    return voronoi.reshape(coordinates.shape[:-1])


def generate_weighted_voronoi(seeds,coordinates,weights,N_neighbours=10,weight_type='additive'):

    # Quick catch for unweighted voronoi
    if np.sum(weights)==0.0:
        return generate_voronoi(seeds,coordinates)

    else:
        # Create the KDTree
        seed_tree = scipy.spatial.KDTree(seeds)
    
        # Find the distance and colors
        distance,colors = seed_tree.query(coordinates.reshape(-1,coordinates.shape[-1]), k=N_neighbours, workers = 8 )
        
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

def calculate_accuracy(voronoi_reference,voronoi_perturbed,layer=0):

    accuracy = (voronoi_reference[:,:,layer]==voronoi_perturbed[:,:,layer])*1
    
    return np.sum(accuracy)/np.multiply(*accuracy.shape)

def get_voronoi_surface(voronoi_matrix):

    shape = voronoi_matrix.shape

    if len(shape) == 2:
        return voronoi_matrix

    elif len(shape) == 3:
        return voronoi_matrix[:,:,0]

def get_grain_centre(voronoi_matrix,seed_ID,size,voxels):

    grain_centre = np.array([np.mean(np.where(voronoi_matrix==ID),axis=1) for ID in seed_ID])

    return grain_centre*(size/voxels)


