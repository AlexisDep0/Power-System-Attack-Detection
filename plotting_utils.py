import matplotlib.pyplot as plt
import os

def plot_scenario(df, title, filename, output_dir, start_time):
    """
    Plots Voltage and Current phases for a given dataframe.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 1. Voltage Plot (Phases A, B, C)
    # Note: We use df['Va'] because 'Va' is a column name in your dataframe
    ax1.plot(df['Time'], df['Va'], label='Phase A', color='tab:blue', linewidth=1.2)
    ax1.plot(df['Time'], df['Vb'], label='Phase B', color='tab:orange', linewidth=1.2)
    ax1.plot(df['Time'], df['Vc'], label='Phase C', color='tab:green', linewidth=1.2)
    
    ax1.set_title(title)
    ax1.set_ylabel('Voltage [V]')
    ax1.set_xlim(start_time - 0.05, start_time + 0.3) # Zooming in
    ax1.grid(True, which='both', linestyle='--', alpha=0.5)
    ax1.legend(loc='upper right')

    # 2. Current Plot (Phases A, B, C)
    ax2.plot(df['Time'], df['Ia'], label='Phase A', color='tab:blue', linewidth=1.2)
    ax2.plot(df['Time'], df['Ib'], label='Phase B', color='tab:orange', linewidth=1.2)
    ax2.plot(df['Time'], df['Ic'], label='Phase C', color='tab:green', linewidth=1.2)
    
    ax2.set_ylabel('Current [A]')
    ax2.set_xlabel('Time [s]')
    ax2.set_xlim(start_time - 0.05, start_time + 0.3)
    ax2.grid(True, which='both', linestyle='--', alpha=0.5)
    ax2.legend(loc='upper right')

    plt.tight_layout()

    # Save Plot
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path)
    plt.close(fig)