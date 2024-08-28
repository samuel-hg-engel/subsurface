from subsurface.imports import *
from subsurface.tools import *
from subsurface.voronoi import *

def voronoi_markov_chain(N_iter,accuracy_threshold,perturbation,voronoi_reference,output_frequency=1,weighted=False):

    voronoi_chain = copy.copy(voronoi_reference)

    successful_voronoi=[]

    for N in range(N_iter):
    
        print('Searching...{N}/{N_max}'.format(N=N,N_max=N_iter) + 'Found...{G}'.format(G=len(successful_voronoi)).rjust(20) ,end='\r')
        #print('Found...{G}'.format(G=len(successful_voronoi)).rjust(15) ,end='\r')
        
        voronoi_perturbed = copy.copy(voronoi_chain)
        
        voronoi_perturbed.perturb_seed_locations(perturbation)

        voronoi_perturbed.generate_voronoi_matrix(weighted=weighted)
    
        accuracy = calculate_accuracy(voronoi_reference.voronoi_matrix,voronoi_perturbed.voronoi_matrix,layer=0)
    
        # If we reconstruct the surface within some error, save the seeds and make them the new chain seed
        if accuracy > accuracy_threshold:
                        
                successful_voronoi.append(voronoi_perturbed)
        
                voronoi_chain = voronoi_perturbed

    return successful_voronoi[::output_frequency]