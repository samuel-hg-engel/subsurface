from subsurface.imports import *
from subsurface.tools import *

class Voronoi:
    def __init__(self,voxels,seeds,size):

        # Inherent properties
        self.voxels = voxels
        self.seeds = seeds
        self.size  = size

        # Derived properties
        self._seed_locations = generate_seeds(self.seeds,self.size)
        self._coordinates = generate_coordinates(self.voxels,self.size)
        self._seed_IDs = None
        self._voronoi_matrix = None
        self._surface = None
        self._grain_centre = None
        self._seed_weights = None


    # Allocatable Variables
    # Seed Locations
    @property
    def seed_locations(self):
        return self._seed_locations

    @seed_locations.setter
    def seed_locations(self,value):
        self._seed_locations = value
        self.seeds = len(value)

    @property
    def seed_weights(self):
        return self._seed_weights

    @seed_weights.setter
    def seed_weights(self,value):
        self._seed_weights = value

    # Dependent variables
    # Voronoi Matrix - create the matrix and scalp the surface
    def generate_voronoi_matrix(self,weighted=False,N_neighbours=10):

        if weighted==False:
            self._voronoi_matrix = generate_voronoi(self._seed_locations,self._coordinates)

        elif weighted==True:
            self._voronoi_matrix = generate_weighted_voronoi(self._seed_locations,self._coordinates,self._seed_weights,N_neighbours)

        return self

    def generate_seed_weights(self,min_weight=0,max_weight=1):

        self._seed_weights = np.random.uniform(min_weight,max_weight,size=(self.seeds))
        #self._seed_weights = max_weight*np.random.random(size=(self.seeds))
        
        return self
    
    @property
    def voronoi_matrix(self):
        return self._voronoi_matrix

    # Voronoi Surface
    @property
    def surface(self):
        self._surface = get_voronoi_surface(self._voronoi_matrix)
        return self._surface

    # Seed IDs
    @property
    def seed_IDs(self):
        self._seed_IDs = np.unique(self._voronoi_matrix)
        return self._seed_IDs

    # Grain centre
    @property
    def grain_centre(self):
        self._seed_IDs = np.unique(self._voronoi_matrix)
        self._grain_centre = get_grain_centre(self._voronoi_matrix,self._seed_IDs,self.size,self.voxels)
        return self._grain_centre

    # Generic Methods
    def save(self,filename):
    
        # Use damask to save the voronoi matrix
        grid = damask.GeomGrid.from_Voronoi_tessellation(size=[1,1,1],seeds=[[1,1,1]],cells=[1,1,1],periodic=False)
        
        grid.size = self.size
        grid.material = self._voronoi_matrix
        grid.save(fname=filename)
        
    def resample_seeds(self):
        self._seed_locations = generate_seeds(self.seeds,self.voxels)
        self._seed_IDs = np.unique(self._voronoi_matrix)
        return self._seed_locations

    def perturb_seed_locations(self,perturbation): 
        self._seed_locations = perturb_seeds(self._seed_locations,perturbation)
        return self._seed_locations

    def perturb_seed_weights(self,perturbation): 
        self._seed_weights = perturb_weights(self._seed_weights,perturbation)
        return self._seed_weights

