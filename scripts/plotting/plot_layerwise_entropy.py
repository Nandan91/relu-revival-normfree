import matplotlib.pyplot as plt
import os
import numpy as np

def read_entropy_data(file_name):
    steps = []
    entropies = []
    with open(file_name, 'r') as file:
        for line in file:
            parts = line.split()
            step = int(parts[1])
            entropy = float(parts[-1])
            steps.append(step)
            entropies.append(entropy)
    return steps, entropies

def format_ticks(x, pos):
    return f'{int(x/1000)}K'

def annotate_layer(ax, steps, entropies, layer_index, text, fontsize=12):
    # Choose the last step for annotation
    step = steps[-30]
    entropy = entropies[-30]
    ax.annotate(text, (step, entropy), textcoords="offset points", xytext=(10, 10), ha='center',fontsize=fontsize)

def plot_entropy():
    plt.figure(figsize=(12, 8))

    # Use a colormap for diverse colors
    colormap = plt.cm.viridis
    # colormap = plt.get_cmap('tab20')
    colors = [colormap(i) for i in np.linspace(0, 1, 12)]

    ax = plt.gca()

    for i in range(12):
        file_name = f"layer_{i}_entropy.txt"
        steps, entropies = read_entropy_data(file_name)
        ax.plot(steps, entropies, label=f'Layer {i}', color=colors[i], linewidth=2)

        # Add annotations for specific layers (e.g., layer 10 and 11) at the end of the line
        if i in [0, 1]:
            annotate_layer(ax, steps, entropies, i, f'Layer {i}', fontsize=20)

    plt.xlabel('Steps', fontsize=30)
    plt.ylabel('Mean Entropy', fontsize=30)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.legend(loc='upper right', fontsize=20, ncol=4)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(format_ticks))
    
    plt.tight_layout()
    plt.savefig('ent_layerwise_SM_LN_G.pdf', format='pdf', dpi=300)

plot_entropy()
