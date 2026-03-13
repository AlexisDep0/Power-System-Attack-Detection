import matlab.engine
import numpy as np
import pandas as pd
#from mimic_attack import apply_sensor_attack #uncomment once other script finished

#start matlab engine
eng = matlab.engine.start_matlab()

#run simulation and catch output as raws
t_raw, v_raw, i_raw = eng.run_custom_fault('1000', 0.01, 0.01, 0.1, 0.2, nargout=3)

eng.quit()

#dataframe structure
df = pd.DataFrame({
    'Time': np.array(t_raw).flatten(),
    'Va': np.array(v_raw)[:,0], 'Vb': np.array(v_raw)[:,1], 'Vc': np.array(v_raw)[:,2],
    'Ia': np.array(i_raw)[:,0], 'Ib': np.array(i_raw)[:,1], 'Ic': np.array(i_raw)[:,2]
})

#mimic sensor attack
#df_attacked = apply_sensor_attack(df, attack_type='sag') #uncomment this line as well

#for learning model training
#df_attacked.to_csv('sim_output.csv', index=False)
print("Physical simulation complete. Data passed to mimic_attack.py.\n")