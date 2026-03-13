import pandas as pd
import numpy as np

def apply_sensor_attack(df, attack_type):
    """
    TEAMMATE B: Use the parameters below to mimic the real faults.
    Real Fault Window: 0.1s to 0.15s
    Real Ron Range: 0.001 to 0.02
    """
    attack_df = df.copy()
    
    # Define the attack window to match the physical simulation
    mask = (attack_df['Time'] >= 0.1) & (attack_df['Time'] <= 0.15)

    if attack_type == 'SLG_mimic':
        # TODO: Manipulate one voltage/current phase to mimic A-G, B-G, or C-G
        pass
        
    elif attack_type == 'LL_mimic':
        # TODO: Manipulate two phases to mimic A-B, B-C, or C-A
        pass

    elif attack_type == 'ThreePhase_mimic':
        # TODO: Manipulate all phases to mimic ABC or ABC-G
        pass

    # TODO: Recalculate Total Power to highlight the "Sensor Lie"
    # attack_df['P_Total'] = ...
    
    return attack_df