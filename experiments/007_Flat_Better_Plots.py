import matplotlib.pyplot as plt
import numpy as np

# Example data
models = [f"Model {i}" for i in range(1, 6)]  # 5 models
plots_per_model = 4

# Create a grid for all models
fig, axes = plt.subplots(len(models), plots_per_model, figsize = (15, 10))
#
# # Plot dummy data
# for i, model in enumerate(models):
#     for j in range(plots_per_model):
#         x = np.linspace(0, 10, 100)
#         y = np.sin(x + (i + 1) * j)
#         axes[i, j].plot(x, y)
#         axes[i, j].set_title(f"{model} - Plot {j + 1}")
#         axes[i, j].tick_params(axis='both', which='major', labelsize=8)
#
# plt.tight_layout()
# plt.show()


for model in models:
    fig, axes = plt.subplots(1, plots_per_model, figsize = (15, 3))
    for j in range(plots_per_model):
        x = np.linspace(0, 10, 100)
        y = np.cos(x + (models.index(model) + 1) * j)
        axes[j].plot(x, y)
        axes[j].set_title(f"{model} - Plot {j + 1}")
    plt.tight_layout()
    plt.savefig(f"{model}_plots.png")
    plt.close(fig)
