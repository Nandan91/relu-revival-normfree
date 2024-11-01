import matplotlib.pyplot as plt
import seaborn as sns
import json
import numpy as np

# Load the data from the log file
with open('input.txt', 'r') as file:
    data = json.load(file)

# Extract entropy values for each layer and head
entropy_values = [[0 for _ in range(12)] for _ in range(12)]  # 12x12 matrix for 12 layers and 12 heads
for key, value in data.items():
    if "attn." in key and "rank" in key:
        parts = key.split('.')
        layer = int(parts[1])
        head = int(parts[3])
        entropy_values[layer][head] = value

# Convert to NumPy array for easier calculations
entropy_array = np.array(entropy_values)

# Calculate mean entropy for each layer and each head
mean_entropy_per_layer = np.mean(entropy_array, axis=1)
mean_entropy_per_head = np.mean(entropy_array, axis=0)

# Adding the mean entropy to the heatmap matrix
extended_entropy = np.append(entropy_array, mean_entropy_per_layer[:, None], axis=1)
extended_entropy = np.append(extended_entropy, np.append(mean_entropy_per_head, np.mean(entropy_array)).reshape(1, -1), axis=0)


plt.figure(figsize=(14, 9.2))
ax = sns.heatmap(entropy_array, annot=True, fmt=".0f", linewidths=.5, cmap='cividis', cbar=True, annot_kws={"size": 22})


# Setting titles and labels with increased font size
# plt.title('Entropy heatmap (Baseline with GELU)', fontsize=20)
plt.xlabel('Head index', fontsize=32)
plt.ylabel('Layer index', fontsize=32)

# Adjusting x-ticks and y-ticks
ax.set_xticks(np.arange(0.5, 12.5, 1))
ax.set_yticks(np.arange(0.5, 12.5, 1))
ax.set_xticklabels(list(range(12)), fontsize=24)
ax.set_yticklabels(list(range(12)), fontsize=24)

cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=22)
plt.tight_layout()

# Save the plot as a high quality PDF
plt.savefig('rank_hmap_SM_LN_G.pdf', format='pdf', dpi=300, bbox_inches='tight')

# Display the plot
plt.show()
