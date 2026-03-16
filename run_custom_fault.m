function [t, v, i] = run_custom_fault(faultCode, Ron, Rg, startTime, endTime)
    modelName = 'CPE_493A_Project';
    blockPath = [modelName '/Three-Phase Fault'];

    load_system(modelName);

    %Define the block path
    faultBlock = [modelName, '/Three-Phase Fault'];

    %Parameter Manipulation for Three-phase Fault Block
    %First set all switches to off
    set_param(faultBlock, 'FaultA', 'off', 'FaultB', 'off', 'FaultC', 'off', 'GroundFault', 'off');
    
    
switch faultCode
        case 'Normal'
            set_param(blockPath, 'FaultA', 'off', 'FaultB', 'off', 'FaultC', 'off', 'GroundFault', 'off');
            % MOVE the fault time far outside the simulation range (e.g., to 99 seconds)
            set_param(blockPath, 'SwitchTimes', '[99 100]'); 
            
        case 'AG'
            set_param(blockPath, 'FaultA', 'on', 'GroundFault', 'on');
            set_param(blockPath, 'SwitchTimes', sprintf('[%f %f]', startTime, endTime));
            
        case 'AB'
            set_param(blockPath, 'FaultA', 'on', 'FaultB', 'on');
            set_param(blockPath, 'SwitchTimes', sprintf('[%f %f]', startTime, endTime));
            
        case 'ABC'
            set_param(blockPath, 'FaultA', 'on', 'FaultB', 'on', 'FaultC', 'on');
            set_param(blockPath, 'SwitchTimes', sprintf('[%f %f]', startTime, endTime));
    end
    
    set_param(faultBlock, 'FaultResistance', num2str(Ron));
    set_param(faultBlock, 'GroundResistance', num2str(Rg));

    %Run simulation
    simOut = sim(modelName, 'ReturnWorkspaceOutputs', 'on');

    t = simOut.tout;
    
    % Check if the data is a Timeseries and extract the numeric 'Data' property
    if isa(simOut.simout, 'matlab.timeseries.Timeseries') || isa(simOut.simout, 'timeseries')
        v = simOut.simout.Data;
    else
        v = simOut.simout; % Already an array
    end
    
    if isa(simOut.simout1, 'matlab.timeseries.Timeseries') || isa(simOut.simout1, 'timeseries')
        i = simOut.simout1.Data;
    else
        i = simOut.simout1; % Already an array
    end
