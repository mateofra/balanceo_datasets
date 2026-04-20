import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import ipywidgets as widgets
from IPython.display import display, clear_output

# MANO connections for skeleton plotting
CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8), # Index
    (0, 9), (9, 10), (10, 11), (11, 12), # Middle
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
]

class ManoValidationUI:
    def __init__(self, data_path, samples_dir):
        self.data_path = data_path
        self.samples_dir = samples_dir
        self.df = pd.read_csv(data_path)
        self.clusters = sorted(self.df['cluster'].unique())
        
        # UI Components
        self.cluster_select = widgets.Dropdown(
            options=self.clusters,
            description='Cluster:',
            value=self.clusters[0]
        )
        self.label_input = widgets.Dropdown(
            options=['fist', 'palm', 'one', 'two_up', 'peace', 'three', 'four', 'rock', 'ok', 'ambiguous'],
            description='New Label:',
            value='palm'
        )
        self.apply_btn = widgets.Button(description='Apply to Cluster', button_style='success')
        self.save_btn = widgets.Button(description='Save CSV', button_style='primary')
        self.output = widgets.Output()
        
        self.apply_btn.on_click(self.on_apply_clicked)
        self.save_btn.on_click(self.on_save_clicked)
        self.cluster_select.observe(self.on_cluster_change, names='value')
        
    def plot_cluster_grid(self, cluster_id, n_samples=64):
        cluster_df = self.df[self.df['cluster'] == cluster_id].head(n_samples)
        n = len(cluster_df)
        cols = 8
        rows = (n + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 2 * rows))
        axes = axes.flatten()
        
        for i, (idx, row) in enumerate(cluster_df.iterrows()):
            ax = axes[i]
            # Load 2D landmarks for plotting (more efficient than 3D for grid)
            # Filename logic from generator.py: sample_00000_MST_1_landmarks.npy
            # sample_id format in df: sample_00000_MST_1
            lm_path = os.path.join(self.samples_dir, f"{row['sample_id']}_landmarks.npy")
            
            if os.path.exists(lm_path):
                lms = np.load(lm_path)
                # Plot skeleton
                for start, end in CONNECTIONS:
                    ax.plot([lms[start, 0], lms[end, 0]], [lms[start, 1], lms[end, 1]], 'b-', alpha=0.6)
                ax.scatter(lms[:, 0], lms[:, 1], s=5, c='r')
            
            ax.set_title(f"{row['suggested_label']}", fontsize=8)
            ax.axis('off')
            ax.set_aspect('equal')
            
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')
            
        plt.tight_layout()
        plt.show()

    def on_cluster_change(self, change):
        with self.output:
            clear_output(wait=True)
            self.plot_cluster_grid(change['new'])

    def on_apply_clicked(self, b):
        cluster_id = self.cluster_select.value
        new_label = self.label_input.value
        self.df.loc[self.df['cluster'] == cluster_id, 'suggested_label'] = new_label
        with self.output:
            print(f"Applied '{new_label}' to cluster {cluster_id}")
            self.on_cluster_change({'new': cluster_id})

    def on_save_clicked(self, b):
        self.df.to_csv(self.data_path, index=False)
        with self.output:
            print(f"Saved changes to {self.data_path}")

    def show(self):
        display(widgets.VBox([
            widgets.HBox([self.cluster_select, self.label_input, self.apply_btn, self.save_btn]),
            self.output
        ]))
        with self.output:
            self.plot_cluster_grid(self.cluster_select.value)

def launch_ui(csv_path="output/mano_assisted_labeling_step_ab.csv", 
              samples_dir="datasets/synthetic_mst/mano_samples_balanced"):
    ui = ManoValidationUI(csv_path, samples_dir)
    ui.show()
