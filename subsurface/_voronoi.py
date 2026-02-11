from subsurface.imports import *
from subsurface.tools import *
from subsurface.construction import *

class Voronoi:
    def __init__(self,
                 voxels: list[int],
                 size: list[float],
                 seeds: list[list[float]],
                 clusters: list[list[float]] | None = None,
                 periodic: bool = False,
                 origin: list[float] = [0.0]*3,
                 minkowski: float = 2.0,
                 ):
        """
        Voronoi Tesselation Generation.

        Parameters
        ----------
        voxels : sequence of ints with length of (3)
            Number of voxels along the x, y, z direction.
        seeds : list[list[float]] 
            Seed locations relative to origin in [m].
        clusters : list[list[float]] 
            Cluster locations relative to the origin in [m].
        size : sequence of floats with length of (3)
            Size in [m] of the edge lengths of the RVE.
        """ 
        self.voxels = voxels
        self.size  = size
        self.seeds = seeds
        self.origin = origin
        self.periodic = periodic
        self.minkowski = minkowski
        self.clusters = clusters

    def __str__(self):

        statement = (f'Voronoi Object\n'
                     f'Seeds : {len(self.seeds)}\n'
                     f'Clusters : {len(self.clusters)} [m]\n'
                     f'Size : {self.size} [m]\n'
                     f'Voxels : {self.voxels}\n'
                     )
        
        return statement

    @property
    def voxels(self) -> np.ndarray:
        """Voxel counts along the x,y,z directions."""
        return self._voxels

    @voxels.setter
    def voxels(self,
               voxels: list[int],
               ):
        
        if (len(voxels) != 3):
            raise ValueError('Voxels must be 3D.')
        
        self._voxels = np.asarray(voxels)
            
    @property
    def size(self) -> np.ndarray:
        """Size of the grid in [m]."""
        return self._size

    @size.setter
    def size(self,
             size: list[float],
             ):
        
        if (len(size) != 3):
            raise ValueError('Size must 3D.')
        
        self._size = np.asarray(size)
    
    @property
    def seeds(self) -> np.ndarray:
        """Seed locations in [m]."""
        return self._seeds
    
    @seeds.setter
    def seeds(self,
              seeds: list[list[float]],
              ):
        self._seeds = np.asarray(seeds)

    @property
    def clusters(self) -> np.ndarray:
        """Cluster locations in [m]."""
        return self._clusters
    
    @clusters.setter
    def clusters(self,
              clusters: list[list[float]],
              ):
        self._clusters = np.asarray(clusters)

    @property
    def origin(self) -> np.ndarray:
        """Origin of the grid in [m]."""
        return self._origin
    
    @origin.setter
    def origin(self,
               origin: list[float]
               ):
        if (len(origin) != 3):
            raise ValueError('Origin must be 3D.')
        self._origin = np.asarray(origin)

    @property
    def periodic(self) -> bool:
        """Periodicity of the tesselation."""
        return self._periodic
    
    @periodic.setter
    def periodic(self,
                 periodic: bool,
                 ):
        self._periodic = periodic
    
    @property
    def minkowski(self) -> float:
        """Minkowski p-norm for generation."""
        return self._minkowski
    
    @minkowski.setter
    def minkowski(self,
                minkowski: float
                ):
        self._minkowski = minkowski

    @property
    def coordinates(self) -> np.ndarray:
        """Coordinates of the tesselation grid."""
        return self._coordinates
    
    @coordinates.setter
    def coordinates(self,
                coordinates):
        self._coordinates = coordinates

    @property
    def matrix(self) -> np.ndarray:
        """The Voronoi tesselation grid."""
        return self._matrix
    
    @matrix.setter
    def matrix(self,
                matrix):
        self._matrix = matrix

    # Derived inputs
    @property
    def surface(self):
        """The surface of the Voronoi tesselation"""
        self._surface = get_voronoi_surface(self._matrix)
        return self._surface

    @property
    def grain_centre(self):
        """The centre of each generated grain."""
        seed_values = np.unique(self.matrix)
        self._grain_centre = get_grain_centre(self._matrix,self._seeds,seed_values,self._size,self._voxels)
        return self._grain_centre

    def generate_coordinates(self):
        """Generate the coordinates of the tesselation grid."""
        
        start = self._origin              + self._size/self._voxels*0.5 # Add voxel centre offset
        end = self._origin + self._size   - self._size/self._voxels*0.5 # Add voxel offset
    
        self._coordinates = np.stack(np.meshgrid(np.linspace(start[0],end[0],self._voxels[0]),
                                           np.linspace(start[1],end[1],self._voxels[1]),
                                           np.linspace(start[2],end[2],self._voxels[2]),
                                           indexing='ij'),
                                           axis=-1)
        return self

    def generate_matrix(self,workers=8):
        """Generate the values of the tesselation grid."""

        seed_tree = scipy.spatial.KDTree(self._seeds,boxsize=self._size if self._periodic else None)

        matrix = seed_tree.query(self._coordinates.reshape(-1,self._coordinates.shape[-1]), workers=workers,p=self._minkowski)[1]
         
        self.matrix = matrix.reshape(self._coordinates.shape[:-1])

        return self

    def generate_clusters(self,workers=4):
        """Cluster the Voronoi cells in the tesselation."""

        cluster_tree = scipy.spatial.KDTree(self._clusters,boxsize=self._size if self._periodic else None)

        seed_cluster = cluster_tree.query(self._seeds,workers=workers)[1]

        cluster_dict = dict(zip(range(len(self._seeds)),seed_cluster))

        self.matrix = np.vectorize(lambda x: cluster_dict[x])(self.matrix)

        return self

    def save(self,filename):
        """Save the tesselation as a VTI."""

        # Generate a dummy grid using damask
        grid = damask.GeomGrid.from_Voronoi_tessellation(size=[1,1,1],seeds=[[1,1,1]],cells=[1,1,1],periodic=False)
        
        grid.size = self._size
        grid.material = self._matrix
        grid.save(fname=filename)

        return self


    def perturb_seed_locations(self,
                               scale: float,
                               bounded: bool = False,
                               weights: np.ndarray| None = None
                               ) -> np.ndarray:
        """

        Method to perturb the seeds in a Voronoi tesselation.

        Parameters
        ----------            
            scale (float): Width the perturbation distribution in [m]. 
            bounded (bool, optional): Whether to use periodic boundary conditions when perturbing. Defaults to False.
            weights (np.ndarray | None, optional): Weight of the x,y,z pertuburbation for each seed. Defaults to None.

        Returns:
            perturbed_seeds (np.ndarray): New seed locations in [m].
        """
        if weights is not None and weights.shape != self._seeds.shape:
            raise AssertionError("Perturbation weights must have the same shape as the seeds.")

        weights = weights if weights is not None else np.ones(self._seeds.shape)

        perturbation = 2*(np.random.random(size=(self._seeds.shape))-0.5)*scale

        self._seeds = self._seeds + perturbation*weights

        if bounded==True:
            self._seeds = np.mod(self._seeds,self._size)

        return self._seeds