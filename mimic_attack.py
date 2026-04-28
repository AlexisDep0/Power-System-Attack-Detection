import pandas as pd
import numpy as np

def apply_sensor_attack(df, attack_type, start_time, stop_time):
    """
    Injects cyber-layer manipulations into a 'Normal' dataset 
    to mimic physical faults without physical consistency.
    """
    attack_df = df.copy()
    
    # Matching the 0.05s window from your teammate's data_collection.py
    # Note: data_collection uses variable start and stop time
    mask = (attack_df['Time'] >= start_time) & (attack_df['Time'] <= stop_time)

    if attack_type == 'SLG_mimic':
        # Single Line-to-Ground: Mimic Phase A voltage dropping to near zero
        # In a real fault, Ia would spike. In this attack, Ia stays normal.
        attack_df.loc[mask, 'Va'] *= 0.05 
        attack_df['Fault_Type'] = 'Attack_SLG'
        
    elif attack_type == 'LL_mimic':
        # Line-to-Line: Mimic Phase A and B voltages dropping and converging
        attack_df.loc[mask, 'Va'] *= 0.5
        attack_df.loc[mask, 'Vb'] *= 0.5
        attack_df['Fault_Type'] = 'Attack_LL'

    elif attack_type == 'ThreePhase_mimic':
        # Balanced Three-Phase: All voltages drop significantly
        for phase in ['Va', 'Vb', 'Vc']:
            attack_df.loc[mask, phase] *= 0.1
        attack_df['Fault_Type'] = 'Attack_ABC'

    # RECALCULATE POWER: This is the "Tell" for the detection script
    # Because V is low but I is normal, P will drop unnaturally.
    attack_df['Pa'] = attack_df['Va'] * attack_df['Ia']
    attack_df['Pb'] = attack_df['Vb'] * attack_df['Ib']
    attack_df['Pc'] = attack_df['Vc'] * attack_df['Ic']
    attack_df['P_Total'] = attack_df['Pa'] + attack_df['Pb'] + attack_df['Pc']
    
    # Binary Label: 1 means something is wrong (either fault or attack)
    attack_df['Attack_Label'] = 1
    
    return attack_df

# QUICK TEST WORKFLOW
if __name__ == "__main__":
    # 1. Load the "Normal" run from your teammate's output folder
    try:
        baseline = pd.read_csv('sim_results_TIMESTAMP/sim_output_combined.csv')
        normal_data = baseline[baseline['Fault_Type'] == 'Normal']
        
        # 2. Generate a mimic attack
        fdi_attack_data = apply_sensor_attack(normal_data, 'SLG_mimic')
        
        # 3. Save for the ML classifier
        fdi_attack_data.to_csv('cyber_attack_dataset.csv', index=False)
        print("Successfully generated cyber attack dataset.")
    except Exception as e:
        print(f"File not found. Make sure to run data_collection.py first! Error: {e}")