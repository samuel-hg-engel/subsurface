from subsurface.imports import *
from subsurface.tools import *

def MarkovChain(N_iter,
                accuracy_threshold,
                perturbation,
                voronoi_reference,
                output_frequency=1,
                weighted=False,
                periodic=False,
                target_rate=None,
                window_size=1000,
                dampening_factor=0.001,
                return_scale=False,
                return_rate=False):
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
    success_list = []
    perturbation_list = []
    rate_list = []

    # Perform N_iter loops
    for N in range(N_iter):

        perturbation_list.append(perturbation)
        
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

            success_list.append(1)
        else:
            success_list.append(0)


        # Find the success rate over the last 1000 perturbations
        success_rate = np.sum(success_list[-1*window_size+1:])/window_size

        rate_list.append(success_rate)

        # Print statements and adapative perturbation
        # Print statement
        statement = 'Searching...{N}/{N_max}'.format(N=N,N_max=N_iter) + 'Success Rate..{X}'.format(X=np.round(success_rate,3)).rjust(20,'.') + '..' +str(np.round(perturbation*1e6,3))

        print(statement, end='\r')

        # If a target rate has been specified then change the perturbation to achieve it
        if not target_rate is None:

            # We look after an initial window_size of increments to make sure things have settled down
            if N>window_size:

                # We only adjust the perturbation if we drift outside 10% of the target rate
                if (success_rate > target_rate*1.1) or (success_rate < target_rate*0.9):

                    # Find the difference between our success rate and the target rate
                    difference = np.sign(target_rate - success_rate)*dampening_factor*((target_rate - success_rate)/target_rate)**2

                    # Scale factor to change the perturbation by
                    scale_factor = 1 - difference

                    # Adjust the perturbation
                    perturbation = perturbation*scale_factor

    if return_scale==True:

        if return_rate==True:

            return successful_voronoi[::output_frequency], perturbation_list[::output_frequency], rate_list[::output_frequency]
    
        elif return_rate==False:

            return successful_voronoi[::output_frequency], perturbation_list[::output_frequency]
        
    elif return_scale==False:

        if return_rate==True:

            return successful_voronoi[::output_frequency], rate_list[::output_frequency]
        
        elif return_rate==False:
            
            return successful_voronoi[::output_frequency]
    
    else:
        return successful_voronoi[::output_frequency]