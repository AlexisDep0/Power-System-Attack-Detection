# Power System Attack Detection Dataset Generator

This repository contains a hybrid **Python-MATLAB-Simulink** toolset designed to generate high-fidelity power system datasets. It simulates various three-phase fault scenarios (Natural vs. Malicious) on a 12.47kV distribution feeder and exports the results as synchronized CSV files and visualizations for machine learning training.

## 🛠 Prerequisites & Versioning

To ensure the MATLAB Engine for Python communicates correctly, you must use these specific versions:

* **Python:** 3.12.8
* **MATLAB / Simulink:** R2024b
* **Required MATLAB Toolboxes:** * Simscape
    * Simscape Electrical (formerly SimPowerSystems)
* **Operating System:** Windows 10/11

## 🚀 Setup Instructions

### 1. Install the MATLAB Engine for Python
This bridge allows Python to "drive" your Simulink model. 
1. Open your terminal (Command Prompt or PowerShell) as an **Administrator**.
2. Navigate to your MATLAB installation folder:
   `cd "C:\Program Files\MATLAB\R2024b\extern\engines\python"`
3. Install the engine:
   ```bash
   python -m pip install .
   ```
### 2. Configure your Python Interpreter
If you have mulitple versions of Python installed, you must ensure your IDE is using **v3.12.8**:
* **VS Code**: Press Crtl+Shift+P, type "Python: Select Interpreter", and select the correct version
* **Verification**: Run `python --version` in your IDE terminal to confirm
### 3. Install Python Dependencies
  ```bash
  pip install numpy pandas matplotlib
  ```
---

## 📂 File Structure
* `data_collection.py`: Main Python script for batch simulation and dataset generation.
* `run_custom_fault.m`: MATLAB wrapper function for parameter injection and fault control.
* `power_system_model.slx`: The Simulink circuit model (12.47kV distribution feeder).
* `/sim_results_[TIMESTAMP]/`: Automatically generated output folder containing:
    * `*_visualization.png`: Event-focused plots for each simulated scenario.
    * `sim_output_combined.csv`: The master training dataset containing all scenarios.

---

## 📊 Data Dictionary (CSV Columns)

| Column | Description | Unit |
| :--- | :--- | :--- |
| **Time** | Simulation timestamp | Seconds ($s$) |
| **Va, Vb, Vc** | Phase-to-Ground RMS Voltage | Volts ($V$) |
| **Ia, Ib, Ic** | Phase RMS Current | Amps ($A$) |
| **Pa, Pb, Pc** | Real Power per phase ($V \times I$) | Watts ($W$) |
| **Fault_Type** | Scenario Label (AG, BC, ABC, Normal) | String |
| **Attack_Label**| Binary Target: 0 (Normal), 1 (Fault/Attack) | Integer |

---

## 🏃 How to Run

1. **Open MATLAB R2024b**: Set the "Current Folder" to this project directory.
2. **File Check**: Ensure `.slx` and `.m` files are visible in the MATLAB file browser.
3. **Headless Mode**: Close the Simulink GUI (the Python script runs the engine in the background to save resources).
4. **Execute**: Run the Python script from your terminal or IDE:
   ```bash
   python data_collection.py
   ```

   ---
   ## ⚠️ Maintenance & Troubleshooting

* **Interpreter Selection**: If the script fails to find `matlab.engine`, verify your IDE is using **Python 3.12.8**. 
    * **VS Code**: Press `Ctrl+Shift+P`, type `Python: Select Interpreter`, and choose the version matching 3.12.8.
    * **PyCharm**: Navigate to `File > Settings > Project > Python Interpreter`.

* **Clean Baseline**: The code is specifically configured to move the `SwitchTimes` of the fault block to `[99 100]` during "Normal" runs. This prevents numerical artifacts or "ghost spikes" at the $t=2s$ mark, ensuring the baseline data is perfectly flat.

* **Fault Window**: Default faults trigger at $t=2s$ and clear at $t=2.05s$. If you need to simulate longer duration faults or change the event window, modify the `start_time` and `stop_time` variables in `data_collection.py`.

* **Simscape Licensing**: Ensure your MATLAB installation includes **Simscape** and **Simscape Electrical**. If these are missing, the `.slx` model will fail to initialize.

* **Path Issues**: If MATLAB reports that it cannot find `run_custom_fault`, ensure that the `.py` script and the `.m` file are in the same directory. The script uses `os.getcwd()` to tell MATLAB where to look for the function.

* **Engine Startup**: The first run may take up to 60 seconds to initialize the MATLAB Engine. Do not terminate the process if it seems "stuck" at the beginning; it is establishing the COM interface.

---
## 🧹 Version Control Note (.gitignore)
To prevent your repository from becoming bloated with simulation results, it is recommended to add the following to your `.gitignore` file:

```text
# Ignore simulation output folders
sim_results_*/
sim_output_combined.csv
*.asv

