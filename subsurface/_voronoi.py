from subsurface.imports import *
from subsurface.tools import *
from subsurface.construction import *

class Voronoi:
    def __init__(self,voxels,seeds,size,weights=None):
        """
        Voronoi Tesselation Generation.

        Parameters
        ----------
        voxels : sequence of ints with length of (3)
            Number of voxels along the x, y, z direction.
        seeds : int
            Number of seeds to generate.
        size : sequence of floats with length of (3)
            Size in [m] of the edge lengths of the RVE.
        weights : Optional sequence of floats of length (seeds)
            Weight of each seed in [m]. This effectively sets the radius of the seed.
        """

        # Inherent properties
        self._voxels = np.array(voxels)
        self._size  = np.array(size)
        self.seeds = int(seeds)
        self.weights = np.array(weights) if weights is not None else None

        # Derived properties
        self._seed_locations = generate_seeds(self.seeds,self._size)
        self._coordinates = generate_coordinates(self._voxels,self._size)
        self._seed_IDs = None
        self._matrix = None
        self._surface = None
        self._grain_centre = None

    def __str__(self):

        statement = (f'Voronoi Object\n'
                     f'Seeds : {self.seeds}\n'
                     f'Size : {self._size} [m]\n'
                     f'Voxels : {self._voxels}\n'
                     f'Generated : {False if self._matrix is None else True}\n'
                     f'Weighted : {False if self.weights is None else True}')

        return statement

    # Allocatable Variables
    # Seed Locations
    @property
    def seed_locations(self):
        return self._seed_locations

    @seed_locations.setter
    def seed_locations(self,value):
        self._seed_locations = value
        self.seeds = len(value)

    # Voxel values
    @property
    def voxels(self):
        return self._voxels

    @voxels.setter
    def voxels(self,value):
        self._voxels = value
        self._coordinates =  generate_coordinates(self._voxels,self._size)
            
    # Size values
    @property
    def size(self):
        return self._size

    @voxels.setter
    def size(self,value):
        self._size = value
        self._coordinates =  generate_coordinates(self._voxels,self._size)
            
    # Dependent Functions
    # Voronoi Matrix - create the matrix and scalp the surface
    def generate_matrix(self,weighted=False,periodic=False,N_neighbours=10):

        if weighted==False:
            self._matrix = generate_voronoi(self._seed_locations,self._coordinates,self._size,periodic=periodic)

        elif weighted==True:
            self._matrix = generate_weighted_voronoi(self._seed_locations,self._coordinates,self._size,self.weights,N_neighbours,periodic=periodic)

        return self

    def generate_weights(self,low=0,high=1):

        self.weights = np.random.uniform(low,high,size=(self.seeds))

        return self
    
    @property
    def matrix(self):
        return self._matrix

    # Voronoi Surface
    @property
    def surface(self):
        self._surface = get_voronoi_surface(self._matrix)
        return self._surface

    # Seed IDs
    @property
    def seed_IDs(self):
        self._seed_IDs = np.unique(self._matrix)
        return self._seed_IDs

    # Grain centre
    @property
    def grain_centre(self):
        self._seed_IDs = np.unique(self._matrix)
        self._grain_centre = get_grain_centre(self._matrix,self._seed_IDs,self._size,self._voxels)
        return self._grain_centre

    # Generic Methods
    def save(self,filename):
    
        # Use damask to save the voronoi matrix
        grid = damask.GeomGrid.from_Voronoi_tessellation(size=[1,1,1],seeds=[[1,1,1]],cells=[1,1,1],periodic=False)
        
        grid.size = self._size
        grid.material = self._matrix
        grid.save(fname=filename)
        
    def resample_seeds(self):
        self._seed_locations = generate_seeds(self.seeds,self._voxels)
        self._seed_IDs = np.unique(self._matrix)
        return self._seed_locations

    def perturb_seed_locations(self,perturbation,periodic=False): 
        self._seed_locations = perturb_seeds(self._seed_locations,perturbation,self._size,periodic)
        return self._seed_locations

    def perturb_weights(self,perturbation): 
        self.weights = perturb_weights(self.weights,perturbation)
        return self.weights

    def add_buffer_layer(self,N_layers=8):
        self._matrix = add_buffer(self._matrix,N_layers)
        return self
