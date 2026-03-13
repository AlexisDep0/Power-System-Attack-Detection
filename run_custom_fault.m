function [t, v, i] = run_custom_fault(faultType, Ron, Rg, startTime, endTime)
    modelName = 'CPE_493A_Project';
    load_system(modelName);

    %Define the block path
    faultBlock = [modelName, '/Three-Phase Fault'];

    %Parameter Manipulation for Three-phase Fault Block
    set_param(faultBlock, 'FaultType', faultType);
    set_param(faultBlock, 'FaultResistance', num2str(Ron));
    set_param(faultBlock, 'GroundResistance', num2str(Rg));
    set_param(faultBlock, 'SwitchTimes', sprintf('[%f %f]', startTime, endTime));

    %Run simulation
    simOut = sim(modelName, 'ReturnWorkspaceOutputs', 'on');

    %Data Extraction direct to Python
    %out.simout(Voltage Vabc) & out.simout1 (Current Iabc)
    t = simOut.out.simout.Time;
    v = simOut.out.simout.Data;
    i = simOut.out.simout1.Data;
end
