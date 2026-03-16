import matlab.engine
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime

#setup folder
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"sim_results_{timestamp}"
os.makedirs(output_dir, exist_ok=True)

#Start MATLAB Engine
print("Starting MATLAB Engine... (this may take 30-60 seconds)")
eng = matlab.engine.start_matlab()

fault_types = ['AG', 'BC','ABC','Normal']
ron = 0.001
rg = 0.01
start_time = 2
stop_time = start_time + 0.05

all_scenarios_data = []


for f_type in fault_types:
    print(f"Running Simulink model for Fault Type: {f_type}...")
    try:
        # Passing 5 arguments now: Type, Ron, Rg, Start, Stop
        t_raw, v_raw, i_raw = eng.run_custom_fault(f_type, float(ron), float(rg), 
                                                   float(start_time), float(stop_time), nargout=3)
        
        # Convert MATLAB data to NumPy
        # MATLAB arrays often come in as list of lists; np.array handles this well
        t = np.array(t_raw).flatten()
        v = np.array(v_raw)
        i = np.array(i_raw)

        # Check if MATLAB returned a column or row vector for V and I
        if v.shape[0] < v.shape[1]: v = v.T
        if i.shape[0] < i.shape[1]: i = i.T

        # Create DataFrame for THIS specific fault
        temp_df = pd.DataFrame({
            'Time': t,
            'Va': v[:, 0], 'Vb': v[:, 1], 'Vc': v[:, 2],
            'Ia': i[:, 0], 'Ib': i[:, 1], 'Ic': i[:, 2],
            'Fault_Type': f_type  # Labeling the data for the team
        })

    # --- NEW: PLOTTING INSIDE THE LOOP ---
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # 1. Voltage Plot
        ax1.plot(t, v[:, 0], label='Phase A', color='blue', linewidth=1.5)
        ax1.set_title(f'Scenario: {f_type} (Fault at {start_time}s)')
        ax1.set_ylabel('RMS Voltage [V]')
        # Focus the view around the fault for the plot
        ax1.set_xlim(start_time - 0.5, start_time + 1.0) 
        ax1.grid(True, which='both', linestyle='--', alpha=0.5)
        ax1.legend()

        # 2. Current Plot
        ax2.plot(t, i[:, 0], label='Phase A', color='red', linewidth=1.5)
        ax2.set_ylabel('RMS Current [A]')
        ax2.set_xlabel('Time [s]')
        ax2.set_xlim(start_time - 0.5, start_time + 1.0)
        ax2.grid(True, which='both', linestyle='--', alpha=0.5)
        ax2.legend()

        plt.tight_layout()

        # Save Plot to Folder
        plot_path = os.path.join(output_dir, f"{f_type}_visualization.png")
        plt.savefig(plot_path)
        plt.close(fig) # Close to free up memory
        
        # Add to master list
        all_scenarios_data.append(temp_df)
        print(f"Scenario {f_type} complete and plot saved.")

    except Exception as e:
        print(f"Error running scenario {f_type}: {e}")
        continue

# Combine and Save
if all_scenarios_data:
    df = pd.concat(all_scenarios_data, ignore_index=True)
    
    # Calculate Power
    df['Pa'] = df['Va'] * df['Ia']
    df['Pb'] = df['Vb'] * df['Ib']
    df['Pc'] = df['Vc'] * df['Ic']
    df['P_Total'] = df['Pa'] + df['Pb'] + df['Pc']

    master_csv_path = os.path.join(output_dir, 'sim_output_combined.csv')
    df.to_csv('sim_output_combined.csv', index=False)
    print("\nMaster dataset saved to 'sim_output_combined.csv'.")

    #Close Engine
    print("Shutting down MATLAB...")
    eng.quit()