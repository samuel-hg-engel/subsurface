import subsurface
from subsurface.imports import *
from subsurface.tools import *


def MarkovChain(voronoi_reference: subsurface.Voronoi,
                N_iter: int,
                accuracy_threshold: float,
                perturbation: float,
                output_frequency: int = 1,
                target_rate: float | None = None,
                window_size: int = 1000,
                dampening_factor: float = 0.001,
                return_scale: bool = False,
                return_rate: bool = False,
                bounded: bool = False,
                layer: int = 0,
                weights: np.ndarray | None = None,
                alpha_scale: float = 1.0,
                chanced: bool = False,
                max_step: float = 1.0
                ):
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
    bounded: bool
        Whether to restrict the seeds to stay within the size of the box.
        Defaults to false.
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
    increment_value = []

    total_displacement = 1e-10

    # Perform N_iter loops
    for N in range(N_iter):

        perturbation_list.append(perturbation)
        
        # Create a perturbed copy
        voronoi_perturbed = copy.copy(voronoi_chain)
        voronoi_perturbed.perturb_seed_locations(perturbation,bounded,weights)
        voronoi_perturbed.generate_matrix()
    
        # Determine the surface accuracy
        accuracy = calculate_accuracy(voronoi_reference.matrix,voronoi_perturbed.matrix,layer=layer)
    
        current_displacement = np.mean(np.linalg.norm(voronoi_reference.seeds-voronoi_perturbed.seeds,axis=1))

        alpha = alpha_scale*(current_displacement / total_displacement)

        if chanced:
            chance = np.random.random(size=1)
        else:
            chance = 1.0
        
        # If we reconstruct the surface within some error, save the seeds and make them the new chain seed
        # We can also check for harsh accuracy. 
        # Firstly check we are maximising the average displacement
        if (alpha > chance) and (accuracy >= accuracy_threshold):
            # update the current best displacement
            total_displacement = current_displacement
                    
            successful_voronoi.append(voronoi_perturbed)

            voronoi_chain = voronoi_perturbed

            success_list.append(1)
            increment_value.append(N)
            
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
                    perturbation = min(max_step,perturbation*scale_factor)

    if return_scale==True:

        if return_rate==True:

            return successful_voronoi[::output_frequency], perturbation_list[::output_frequency], rate_list[::output_frequency],increment_value
    
        elif return_rate==False:

            return successful_voronoi[::output_frequency], perturbation_list[::output_frequency],increment_value
        
    elif return_scale==False:

        if return_rate==True:

            return successful_voronoi[::output_frequency], rate_list[::output_frequency],increment_value
        
        elif return_rate==False:
            
            return successful_voronoi[::output_frequency],increment_value
    
    else:
        return successful_voronoi[::output_frequency], increment_value