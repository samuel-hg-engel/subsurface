from subsurface.imports import *
from subsurface.tools import *

def MarkovChain(N_iter,accuracy_threshold,perturbation,voronoi_reference,output_frequency=1,weighted=False,periodic=False):
    """
    Function to generate a list of perturbed Voronoi objects.

    Parameters
    ----------
    N_iter : int
        Number of iterations to run the markov chain for.
    accuracy_threshold : float
        Fraction of the surface that must be kept constant during permutation.
    perturbation : float
        Perturbation length scale in [m].
    voronoi_reference : Voronoi Object
        A specific Voronoi class object to be perturbed.
    output_frequency : int
        Frequency to save the perturbed Voronoi objects.
    weighted : Bool
        Whether or not the Voronoi object should use the weighted calculation. 
    periodic: Bool
        Whether or not the Voronoi object should adhere to periodic boundary conditions.

    Returns
    -------
    successful_voronoi : list
        List of perturbed Voronoi objects.
    
    """

    # Copy the input voronoi to become an instance for the chain
    voronoi_chain = copy.copy(voronoi_reference)

    # Create a list to hold successful Voronoi
    successful_voronoi=[]

    # Perform N_iter loops
    for N in range(N_iter):

        # Print statement
        statement = 'Searching...{N}/{N_max}'.format(N=N,N_max=N_iter) +  'Found...{G}'.format(G=len(successful_voronoi)).rjust(20,'.')
        print(statement, end='\r')
        
        # Create a perturbed copy
        voronoi_perturbed = copy.copy(voronoi_chain)
        voronoi_perturbed.perturb_seed_locations(perturbation,periodic=periodic)
        voronoi_perturbed.generate_matrix(weighted=weighted,periodic=periodic)
    
        # Determine the surface accuracy
        accuracy = calculate_accuracy(voronoi_reference.matrix,voronoi_perturbed.matrix,layer=0)
    
        # If we reconstruct the surface within some error, save the seeds and make them the new chain seed
        if accuracy > accuracy_threshold:
                        
                successful_voronoi.append(voronoi_perturbed)
        
                voronoi_chain = voronoi_perturbed

    return successful_voronoi[::output_frequency]