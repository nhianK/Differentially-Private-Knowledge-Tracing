import pickle
import os
import matplotlib.pyplot as plt

def load_epsilons(model_name, dataset_name, accountant):
    ckpt_path = os.path.join("ckpts", model_name, dataset_name)
    file_path = os.path.join(ckpt_path, f"epsilons_{accountant}.pkl")
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            epsilons = pickle.load(f)
        print(f"{accountant.upper()} epsilons loaded: {len(epsilons)} values")
        return epsilons
    else:
        print(f"Warning: {file_path} not found.")
        return []

def visualize_epsilons(model_name, dataset_name):
    # Load epsilons for each accountant
    prv_epsilons = load_epsilons(model_name, dataset_name, "prv")
    rdp_epsilons = load_epsilons(model_name, dataset_name, "rdp")
    gdp_epsilons = load_epsilons(model_name, dataset_name, "gdp")

    # Find the maximum number of epochs
    max_epochs = max(len(prv_epsilons), len(rdp_epsilons), len(gdp_epsilons))

    if max_epochs == 0:
        print("Error: No data available for plotting.")
        return

    # Create the plot
    plt.figure(figsize=(12, 6))

    # Plot available data
    if prv_epsilons:
        plt.plot(range(1, len(prv_epsilons) + 1), prv_epsilons, label='PRV', marker='o')
    if rdp_epsilons:
        plt.plot(range(1, len(rdp_epsilons) + 1), rdp_epsilons, label='RDP', marker='s')
    if gdp_epsilons:
        plt.plot(range(1, len(gdp_epsilons) + 1), gdp_epsilons, label='GDP', marker='^')

    plt.title(f'Privacy Budget (Epsilon) over Epochs\n{model_name} on {dataset_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Epsilon')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # Improve x-axis ticks for better readability
    if max_epochs > 20:
        plt.xticks(range(0, max_epochs + 1, max(1, max_epochs // 10)))
    else:
        plt.xticks(range(1, max_epochs + 1))

    # Save the plot
    save_path = os.path.join("ckpts", model_name, dataset_name, "epsilon_comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Visualization saved to {save_path}")

# Example usage
model_name = "dkt"
dataset_name = "Algebra2005"
visualize_epsilons(model_name, dataset_name)