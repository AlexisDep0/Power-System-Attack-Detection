import matlab.engine
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime

from mimic_attack import apply_sensor_attack
from plotting_utils import plot_scenario  

# Setup folder
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"sim_results_{timestamp}" 
os.makedirs(output_dir, exist_ok=True)

# Start MATLAB Engine
print("Starting MATLAB Engine... (this may take 30-60 seconds)")
eng = matlab.engine.start_matlab()

# Simulation Parameters
fault_types = ['AG', 'BC', 'ABC', 'Normal']
ron = 5
rg = 10
start_time = 8
stop_time = start_time + 0.8

all_scenarios_data = []

for f_type in fault_types:
    print(f"Running Simulink model for Fault Type: {f_type}...")
    try:
        t_raw, v_raw, i_raw = eng.run_custom_fault(f_type, float(ron), float(rg), 
                                                   float(start_time), float(stop_time), nargout=3)
        
        t_full = np.array(t_raw).flatten()
        v_full = np.array(v_raw)
        i_full = np.array(i_raw)

        if v_full.shape[0] < v_full.shape[1]: v_full = v_full.T
        if i_full.shape[0] < i_full.shape[1]: i_full = i_full.T

        sample_step = 5 
        t = t_full[::sample_step]
        v = v_full[::sample_step, :]
        i = i_full[::sample_step, :]

        # Create DataFrame with Initial Labels
        temp_df = pd.DataFrame({
            'Time': t,
            'Va': v[:, 0], 'Vb': v[:, 1], 'Vc': v[:, 2],
            'Ia': i[:, 0], 'Ib': i[:, 1], 'Ic': i[:, 2],
            'Fault_Type': f_type,
            'attack_label': 0 if f_type == 'Normal' else 1
        })

        plot_scenario(temp_df, f"Original: {f_type}", f"sim_{f_type}.png", output_dir, start_time)
        
        all_scenarios_data.append(temp_df)
        print(f"Scenario {f_type} complete.")

    except Exception as e:
        print(f"Error running scenario {f_type}: {e}")
        continue

# --- PROCESSING BLOCK ---
if all_scenarios_data:
    df = pd.concat(all_scenarios_data, ignore_index=True)
    attack_datasets = []

    # GENERATE NORMAL FDIA (Label 2)
    normal_data = df[df['attack_label'] == 0].copy()
    if not normal_data.empty:
        print("Generating mimic attack data on Normal scenarios...")
        attack_types = ['SLG_mimic', 'LL_mimic', 'ThreePhase_mimic', 'Drift']
        for attack in attack_types:
            attacked_df = apply_sensor_attack(normal_data, attack, start_time, stop_time)
            plot_scenario(attacked_df, f"Attack: {attack} on Normal", f"attack_2_Normal_{attack}.png", output_dir, start_time)
            attack_datasets.append(attacked_df)

    # GENERATE FAULT FDIA (Label 3)
    fault_data = df[df['attack_label'] == 1].copy()
    if not fault_data.empty:
        print("Generating FDIA on Fault scenarios...")
        fault_attack_types = ['Fault_Mask', 'Fault_Exaggerate']
        
        for f_type in fault_data['Fault_Type'].unique():
            specific_f_df = fault_data[fault_data['Fault_Type'] == f_type].copy()
            for attack in fault_attack_types:
                attacked_f_df = apply_sensor_attack(specific_f_df, attack, start_time, stop_time)
                plot_scenario(attacked_f_df, f"Attack: {attack} on {f_type}", f"attack_3_{f_type}_{attack}.png", output_dir, start_time)
                attack_datasets.append(attacked_f_df)

    # --- FINAL COMBINATION AND CSV SAVING ---
    # This block is now properly aligned to run after all loops finish
    if attack_datasets:
        df = pd.concat([df] + attack_datasets, ignore_index=True)
        print("--- LABEL CHECK ---")
        print(df['attack_label'].value_counts())
    else:
        print("WARNING: No attack datasets were generated!")

    print("Computing final power and unbalance features...")
    df['Pa'] = df['Va'] * df['Ia']
    df['Pb'] = df['Vb'] * df['Ib']
    df['Pc'] = df['Vc'] * df['Ic']
    df['P_Total'] = df['Pa'] + df['Pb'] + df['Pc']

    v_cols = ['Va', 'Vb', 'Vc']
    df['V_Unbalance'] = df[v_cols].max(axis=1) - df[v_cols].min(axis=1)

    print("Computing rolling variance features...")
    df['Va_Var'] = df.groupby(['Fault_Type', 'attack_label'])['Va'].transform(lambda x: x.rolling(5).var()).fillna(0)
    df['Ia_Var'] = df.groupby(['Fault_Type', 'attack_label'])['Ia'].transform(lambda x: x.rolling(5).var()).fillna(0)

    # Save to CSV
    csv_filename = os.path.join(output_dir, 'sim_output_combined.csv')
    df.to_csv(csv_filename, index=False)
    print(f"\nSUCCESS: Master dataset saved to {csv_filename}")

# Close Engine
print("Shutting down MATLAB...")
eng.quit()