# Module imports
from src.imports import *
from src.tools import *
from src.voronoi import *
from src.markovchain import *

# Input parameters
N_iter=10000
threshold = 0.95
perturbation=0.1
output_frequency=1

voxel_size = [100,100,100]
N_seeds = 250

# Generate the reference voronoi object
voronoi_reference = Voronoi(voxel_size,N_seeds) # Optionally can specifiy the seeds

# Generate the voronoi matrix
voronoi_reference.generate_voronoi_matrix # Seeds are autogenerated as they are cheap, matrix must be requested

# Create a list of perturbed voronoi objects
voronoi_objects = voronoi_markov_chain(N_iter,threshold,perturbation,voronoi_reference,output_frequency)

# Save the created objects
for index,voronoi in enumerate(voronoi_objects):

    voronoi.save(filename='results/object_{}.vti'.format(index))

