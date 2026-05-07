% Save Simulink cooling-station results for Python post-processing.
% Usage:
%   out = sim('guanwangmoxing3_2025a');
%   save_sim_result

result_file = fullfile('D:\minicondadaima\lianxi', 'sim_result.mat');

signal_names = {
    'T_ret_3'
    'T_ret_4'
    'T_ret_6'
    'T_sup_3'
    'T_sup_4'
    'T_sup_6'
    'real_flow_3'
    'real_flow_4'
    'real_flow_6'
    'T_station_supply'
    'T_station_return'
    'Q_station_cooling'
    'Q_cooling_removed'
    'Q_cool'
};

sim_result = struct();
sim_result.tout = extract_signal('tout');

for i = 1:numel(signal_names)
    name = signal_names{i};
    value = extract_signal(name);
    if ~isempty(value)
        sim_result.(name) = value;
    end
end

if ~isfield(sim_result, 'Q_station_cooling')
    if isfield(sim_result, 'Q_cooling_removed')
        sim_result.Q_station_cooling = sim_result.Q_cooling_removed;
    elseif isfield(sim_result, 'Q_cool')
        sim_result.Q_station_cooling = sim_result.Q_cool;
    end
end

save(result_file, '-struct', 'sim_result', '-v7');
fprintf('Saved Simulink result to: %s\n', result_file);
disp('Saved variables:');
disp(fieldnames(sim_result));

function value = extract_signal(name)
    value = [];

    if evalin('base', sprintf('exist(''%s'', ''var'')', name))
        value = evalin('base', name);
        value = normalize_signal(value);
        return;
    end

    if evalin('base', 'exist(''out'', ''var'')')
        out_obj = evalin('base', 'out');
        try
            value = out_obj.(name);
            value = normalize_signal(value);
            return;
        catch
        end

        try
            logsout = out_obj.logsout;
            element = logsout.get(name);
            value = normalize_signal(element.Values);
            return;
        catch
        end
    end
end

function value = normalize_signal(value)
    if isa(value, 'timeseries')
        value = value.Data;
    elseif isa(value, 'timetable')
        value = value.Variables;
    elseif isstruct(value) && isfield(value, 'signals')
        value = value.signals.values;
    end

    if isnumeric(value)
        value = squeeze(value);
    end
end
