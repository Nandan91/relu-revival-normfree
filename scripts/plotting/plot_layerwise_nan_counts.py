import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter

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

def annotate_layer(ax, steps, entropies, text, fontsize=30):
    step = steps[0]
    entropy = entropies[30]
    ax.annotate(text, (step, entropy), textcoords="offset points", xytext=(410, 40), 
                ha='center', fontsize=fontsize, rotation=90)

def plot_entropy(use_individual_colors=False):
    plt.figure(figsize=(12, 8))
    if use_individual_colors:
        colors = ['teal', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta', 'blue']
    else:
        colormap = plt.cm.inferno
        colors = [colormap(i) for i in np.linspace(0, 1, 12)]
    
    ax = plt.gca()
    
    for i in range(12):
        file_name = f"layer_{i}_nan_count.txt"
        steps, entropies = read_entropy_data(file_name)
        ax.plot(steps, entropies, label=f'L{i}', color=colors[i], linewidth=2)
        # if i == 11:
        #     annotate_layer(ax, steps, entropies, "Layer11", fontsize=30)

    plt.xlabel('Steps', fontsize=30)
    plt.ylabel('NaN Count', fontsize=30)
    plt.xticks(fontsize=32)
    plt.yticks(fontsize=32)
    # plt.legend(loc='upper left', fontsize=26, ncol=1)

    # Set scientific notation for y-axis
    formatter = ScalarFormatter(useOffset=False, useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))  # Force scientific notation
    ax.yaxis.set_major_formatter(formatter)
    
    # Adjust the scientific notation's exponent font size
    ax.yaxis.get_offset_text().set_fontsize(30)
    
    # Format x-axis
    ax.xaxis.set_major_formatter(plt.FuncFormatter(format_ticks))
    
    plt.tight_layout()
    plt.savefig('layerwise_NaNs_NegSlope1e1.pdf', format='pdf', dpi=300)

plot_entropy(use_individual_colors=True)