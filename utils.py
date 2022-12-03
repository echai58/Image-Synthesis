import matplotlib.pyplot as plt
import numpy as np
import torch

def load_model(atk, pt_file):

    print('Loading checkpoint')

    checkpoint = torch.load(pt_file)
    module_state_dict = checkpoint['state_dict']
    state_dict = {k.replace('module.', ''):v for k,v in module_state_dict.items()}
    atk.load_state_dict(state_dict)

    print(f"Loaded checkpoint at epoch {checkpoint['epoch']}")

    return atk


def plot_all_samples(concat_samples, row_titles, r = 2, c = 10, figsize = (25, 7.5), fontsize = 24, tick_marks = False):
    fig, ax = plt.subplots(r, c, figsize=figsize)

    #For each image
    for i in range(r):
        #For each example
        ax[i][0].set_ylabel(row_titles[i], fontsize=fontsize)
    
        for j in range(len(concat_samples[0])):
            ax_ij = ax[i][j]
    
            current_sample = np.array(concat_samples[i][j])
            #Reshape for imshow
            sample_for_imshow = np.moveaxis(current_sample, source = [0, 1, 2], destination = [-1, -3, -2])
            ax_ij.imshow(sample_for_imshow)
    
        if not tick_marks:
            plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[]);
    
    plt.show()