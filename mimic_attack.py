import pandas as pd
import numpy as np

def apply_sensor_attack(df, attack_type, start_time, stop_time):
    attack_df = df.copy()

    # Ensure base labels exist
    if 'attack_label' not in attack_df.columns:
        attack_df['attack_label'] = np.where(attack_df['Fault_Type'] == 'Normal', 0, 1)

    # Time mask
    mask = (attack_df['Time'] >= start_time) & (attack_df['Time'] <= stop_time)
    idx = attack_df.loc[mask].index

    if len(idx) == 0:
        return attack_df

    # Smooth ramp
    max_perturb = np.random.uniform(0.1, 0.35)
    ramp = np.linspace(0, max_perturb, len(idx))

    # Phase definitions
    V_phases = ['Va', 'Vb', 'Vc']
    I_phases = ['Ia', 'Ib', 'Ic']

    # Detect if input already contains a fault
    # .any() boils the whole column down to a single True or False
    is_fault_data = (attack_df.loc[idx, 'attack_label'] == 1).any()

    #FDIA logic
    if attack_type == 'SLG_mimic':
        p = np.random.randint(0, 3)
        attack_df.loc[idx, V_phases[p]] *= (1 - ramp)
        attack_df.loc[idx, I_phases[p]] *= (1 - 0.3 * ramp)

    elif attack_type == 'LL_mimic':
        phases = np.random.choice([0,1,2], size=2, replace=False)
        for p in phases:
            attack_df.loc[idx, V_phases[p]] *= (1 - ramp)
            attack_df.loc[idx, I_phases[p]] *= (1 - 0.4 * ramp)

    elif attack_type == 'ThreePhase_mimic':
        for v, i in zip(V_phases, I_phases):
            attack_df.loc[idx, v] *= (1 - ramp)
            attack_df.loc[idx, i] *= (1 - 0.5 * ramp)

    elif attack_type == 'Drift':
        drift = np.cumsum(np.random.normal(0, 0.0005, len(idx)))
        for v in V_phases:
            attack_df.loc[idx, v] += drift

    elif attack_type == 'Fault_Mask':
        for v, i in zip(V_phases, I_phases):
            attack_df.loc[idx, v] *= (1 + ramp)
            attack_df.loc[idx, i] *= (1 - 0.4 * ramp)

    elif attack_type == 'Fault_Exaggerate':
        for v, i in zip(V_phases, I_phases):
            attack_df.loc[idx, v] *= (1 - ramp)
            attack_df.loc[idx, i] *= (1 + 0.6 * ramp)

    # Label assignment
    if is_fault_data:
        attack_df.loc[idx, 'attack_label'] = 3   # Fault + FDIA
    else:
        attack_df.loc[idx, 'attack_label'] = 2   # FDIA only

    # ---------------------------
    # Recompute power
    # ---------------------------
    attack_df['Pa'] = attack_df['Va'] * attack_df['Ia']
    attack_df['Pb'] = attack_df['Vb'] * attack_df['Ib']
    attack_df['Pc'] = attack_df['Vc'] * attack_df['Ic']
    attack_df['P_Total'] = attack_df['Pa'] + attack_df['Pb'] + attack_df['Pc']

    return attack_df