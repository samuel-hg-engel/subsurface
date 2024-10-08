from subsurface.imports import *

def generate_coordinates(N_voxels,size,origin= np.zeros(3)):

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

def generate_voronoi(seeds,coordinates,size,periodic=False):
    
    # Create the seed tree
    if periodic == True:
        seed_tree = scipy.spatial.KDTree(seeds,boxsize=size)
    else:
        seed_tree = scipy.spatial.KDTree(seeds)

    voronoi = seed_tree.query(coordinates.reshape(-1,coordinates.shape[-1]), workers = 8 )[1]

    return voronoi.reshape(coordinates.shape[:-1])


def generate_weighted_voronoi(seeds,coordinates,size,weights,N_neighbours=10,weight_type='additive',periodic=False):

    # Quick catch for unweighted voronoi
    if np.sum(weights)==0.0:
        return generate_voronoi(seeds,coordinates,size,periodic)

    else:
        # Create the seed tree
        if periodic == True:
            seed_tree = scipy.spatial.KDTree(seeds,boxsize=size)
        else:
            seed_tree = scipy.spatial.KDTree(seeds)
    
        distance,colors = seed_tree.query(coordinates.reshape(-1,coordinates.shape[-1]), k=N_neighbours, workers=8)

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


def perturb_seeds(seeds,scale,boxsize,periodic=False):

    new_seeds = np.array([np.random.normal(loc=seed,scale=scale) for seed in seeds])

    if periodic==False:
        return  new_seeds

    if periodic==True:
        return np.mod(new_seeds,boxsize)

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

    return grain_centre*np.divide(size,voxels)


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

    
def add_buffer(matrix,N_layers=8):

    x,y,z = matrix.shape

    new_matrix = np.zeros_like(matrix)

    buffer_id = np.max(np.unique(matrix)) + int(1)

    for z_i in range(z):

        layer = matrix[:,:,z_i]

        new_layer = buffer_id*np.ones_like(layer)

        if z_i > (z - N_layers):

            new_matrix[:,:,z_i] = new_layer

        else:
            new_matrix[:,:,z_i] = layer

    return new_matrix


def plot_images(images,titles):

    fig,axes=plt.subplots(1,len(images),figsize=(2*len(images),2))
    
    for i,image in enumerate(images):

        axes[i].imshow(image)

        axes[i].set_xticks([])
        axes[i].set_yticks([])
        axes[i].set_title(titles[i])

    plt.subplots_adjust(wspace=0.05, hspace=0)

    return fig,axes


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