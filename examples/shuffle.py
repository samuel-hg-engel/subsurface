from subsurface.imports import *

from subsurface.tools import plot_images,extract_surfaces
from subsurface import Voronoi
from subsurface import Shuffle

# Generate voronoi
print('Generating Voronoi')
N_seeds = 500
N_voxels = [128]*3
size = np.ones(3)
voronoi = Voronoi(N_voxels,N_seeds,size)
voronoi.generate_matrix()

# Generate orientations
orientations = damask.Rotation.from_random(N_seeds).as_quaternion()

# extract the ID's of the surface grains and internal grains.
surface_grains = np.unique(voronoi.surface)
internal_grains = [int(x) for x in np.unique(voronoi.matrix) if x not in surface_grains]

# This generates the final misorientation distribution.
print('Shuffling Orientations')
orientations_shuffled,misorientation = Shuffle(voronoi.matrix,orientations,iterations=100000,exclude=surface_grains,minimize=True,return_full=True)

# Plot the IPF Colors as a map.
IPF_colors = damask.Orientation(family='cubic',lattice='cI',rotation=damask.Rotation().from_quaternion(orientations_shuffled)).IPF_color(vector=[1,0,0])

# Build a seed ID -> IPF dictionary
IPF_map = dict(zip(np.unique(voronoi.matrix),IPF_colors))

# Find the weights of each nearby seed
orientation_matrix = np.array([IPF_map[i] for i in voronoi.matrix.flatten()]).reshape(*voronoi.matrix.shape,3)

# Plot the results
fig,ax=plot_images(extract_surfaces(orientation_matrix),['-X','+X','-Y','+Y','-Z','+Z'])

# Save the figure
fig.savefig(fname='IPF_map.png')