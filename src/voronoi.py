from src.imports import *
from src.tools import *

class Voronoi:
    def __init__(self,voxels,seeds):

        # Inherent properties
        self.voxels = voxels
        self.seeds = seeds

        # Derived properties
        self._seed_locations = generate_seeds(self.seeds,self.voxels)
        self._coordinates = generate_coordinates(self.voxels)
        self._seed_IDs = None
        self._voronoi_matrix = None
        self._surface = None
        self._grain_centre = None


    # Allocatable Variables
    # Seed Locations
    @property
    def seed_locations(self):
        return self._seed_locations

    @seed_locations.setter
    def seed_locations(self,value):
        self._seed_locations = value
        self.seeds = len(value)

    # Dependent variables
    # Voronoi Matrix - create the matrix and scalp the surface
    @property
    def generate_voronoi_matrix(self):
        self._voronoi_matrix = generate_voronoi(self._seed_locations,self.voxels,self._coordinates)
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
        self._grain_centre = get_grain_centre(self._voronoi_matrix,self._seed_IDs)
        return self._grain_centre

    # Generic Methods
    def save(self,filename):
        numpyToVTK(self._voronoi_matrix,filename)

    def resample_seeds(self):
        self._seed_locations = generate_seeds(self.seeds,self.voxels)
        self._seed_IDs = np.unique(self._voronoi_matrix)
        return self._seed_locations

    def perturb_seed_locations(self,perturbation): 
        self._seed_locations = perturb_seeds(self._seed_locations,perturbation)
        return self._seed_locations

