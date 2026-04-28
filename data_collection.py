import matlab.engine
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime

from mimic_attack import apply_sensor_attack

# Setup folder
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"sim_results_{timestamp}"
os.makedirs(output_dir, exist_ok=True)

# Start MATLAB Engine
print("Starting MATLAB Engine... (this may take 30-60 seconds)")
eng = matlab.engine.start_matlab()

# all can be changed for different simulations
fault_types = ['AG', 'BC', 'ABC', 'Normal']
ron = 0.001
rg = 0.01
start_time = 2
stop_time = start_time + 0.05

all_scenarios_data = []

for f_type in fault_types:
    print(f"Running Simulink model for Fault Type: {f_type}...")
    try:
        # Passing 5 arguments: Type, Ron, Rg, Start, Stop
        t_raw, v_raw, i_raw = eng.run_custom_fault(f_type, float(ron), float(rg), 
                                                   float(start_time), float(stop_time), nargout=3)
        
        # Convert MATLAB data to NumPy
        t_full = np.array(t_raw).flatten()
        v_full = np.array(v_raw)
        i_full = np.array(i_raw)

        # Ensure columns represent phases
        if v_full.shape[0] < v_full.shape[1]: v_full = v_full.T
        if i_full.shape[0] < i_full.shape[1]: i_full = i_full.T

        sample_step = 5 
        t = t_full[::sample_step]
        v = v_full[::sample_step, :]
        i = i_full[::sample_step, :]

        # Create DataFrame
        temp_df = pd.DataFrame({
            'Time': t,
            'Va': v[:, 0], 'Vb': v[:, 1], 'Vc': v[:, 2],
            'Ia': i[:, 0], 'Ib': i[:, 1], 'Ic': i[:, 2],
            'Fault_Type': f_type
        })

        # --- UPDATED: PLOTTING ALL THREE PHASES ---
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # 1. Voltage Plot (Phases A, B, C)
        ax1.plot(t, v[:, 0], label='Phase A', color='tab:blue', linewidth=1.2)
        ax1.plot(t, v[:, 1], label='Phase B', color='tab:orange', linewidth=1.2)
        ax1.plot(t, v[:, 2], label='Phase C', color='tab:green', linewidth=1.2)
        
        ax1.set_title(f'Scenario: {f_type} (Fault at {start_time}s)')
        ax1.set_ylabel('Voltage [V]')
        ax1.set_xlim(start_time - 0.05, start_time + 0.1) # Zoomed in closer to the fault
        ax1.grid(True, which='both', linestyle='--', alpha=0.5)
        ax1.legend(loc='upper right')

        # 2. Current Plot (Phases A, B, C)
        ax2.plot(t, i[:, 0], label='Phase A', color='tab:blue', linewidth=1.2)
        ax2.plot(t, i[:, 1], label='Phase B', color='tab:orange', linewidth=1.2)
        ax2.plot(t, i[:, 2], label='Phase C', color='tab:green', linewidth=1.2)
        
        ax2.set_ylabel('Current [A]')
        ax2.set_xlabel('Time [s]')
        ax2.set_xlim(start_time - 0.05, start_time + 0.1)
        ax2.grid(True, which='both', linestyle='--', alpha=0.5)
        ax2.legend(loc='upper right')

        plt.tight_layout()

        # Save Plot
        plot_path = os.path.join(output_dir, f"{f_type}_visualization.png")
        plt.savefig(plot_path)
        plt.close(fig) 
        
        all_scenarios_data.append(temp_df)
        print(f"Scenario {f_type} complete and plot saved.")

    except Exception as e:
        print(f"Error running scenario {f_type}: {e}")
        continue

# Combine and Save
if all_scenarios_data:
    df = pd.concat(all_scenarios_data, ignore_index=True)

    normal_data = df[df['Fault_Type'] == 'Normal'].copy()

    attack_datasets = []

    if not normal_data.empty:
        print("Generating mimic attack data...\n")
        attack_types = ['SLG_mimic', 'LL_mimic', 'ThreePhase_mimic']

        for attack in attack_types:
            attacked_df = apply_sensor_attack(normal_data, attack, start_time, stop_time)
            attack_datasets.append(attacked_df)

        # Combine all attacks
        attack_df = pd.concat(attack_datasets, ignore_index=True)

        # Merge with original dataset
        df = pd.concat([df, attack_df], ignore_index=True)

        print("Cyber-attack scenarios added to dataset.")
    
    # Calculate Power
    df['Pa'] = df['Va'] * df['Ia']
    df['Pb'] = df['Vb'] * df['Ib']
    df['Pc'] = df['Vc'] * df['Ic']
    df['P_Total'] = df['Pa'] + df['Pb'] + df['Pc']

    #Phase Imbalance (V_diff)
    v_cols = ['Va', 'Vb', 'Vc']
    df['V_Max'] = df[v_cols].max(axis=1)
    df['V_Min'] = df[v_cols].min(axis=1)
    df['V_Unbalance'] = df['V_Max'] - df['V_Min']

    # Moving Variance
    df['Va_Var'] = df['Va'].rolling(window=5).var().fillna(0)
    df['Ia_Var'] = df['Ia'].rolling(window=5).var().fillna(0)

    df.to_csv(os.path.join(output_dir, 'sim_output_combined.csv'), index=False)
    print(f"\nMaster dataset saved to {output_dir}/sim_output_combined.csv")

# Close Engine
print("Shutting down MATLAB...")
eng.quit()