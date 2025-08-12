import pandas as pd
import shared_state as ST
import topology_discovery  # provides gridtopology function
import data_preparation    # provides data_preparation function
import sweep_procedures    # provides forward_backward_sweep function
import print_results       # provides results function

def powerflow(input: str = "", output: str = "",
              tolerance: float = 1e-6, max_iterations: int = 30,
              display_summary: bool = True, timestamp: bool = False,
              display_topology: bool = False, save_topology: bool = False,
              graph_title: str = "", marker_size: float = 1.5, verbose: int = 0):
    """
    Evaluate bus voltage levels and power flows in a three-phase distribution network.
    Reads grid data from CSV files, discovers network topology, performs forward-backward
    sweep iterations, and writes results to CSV files. Returns an error message string
    if execution is aborted; otherwise, returns None (and outputs are saved to files).
    """
    # 1. Data input and topology discovery
    err_msg = topology_discovery.gridtopology(caller="powerflow", input_dir=input, output_dir=output,
                                             timestamp=timestamp, save_topology=save_topology,
                                             display_topology=display_topology,
                                             graph_title=graph_title, marker_size=marker_size,
                                             verbose=verbose)
    if err_msg:
        # Abort if any error during grid topology discovery (e.g., missing files)
        return f"Execution aborted, {err_msg}"

    # 2. Data preparation (set up impedances, loads, generation data, etc.)
    err_msg = data_preparation.data_preparation()
    if err_msg:
        print(f"Execution aborted, {err_msg}")
        return  # abort on data preparation error

    # 3. Power flow iterative solution (outer loop for DG adjustments)
    ST.state.outer_iteration = 0
    max_diff = 1.0
    while max_diff > tolerance:
        # Forward-backward sweep (inner power flow iterations)
        err_msg, inner_iter = sweep_procedures.forward_backward_sweep(tolerance, max_iterations)
        ST.state.inner_iteration = inner_iter  # record latest inner iteration count
        if err_msg:
            # Abort if forward-backward sweep failed or max_iterations reached
            print(f"Execution aborted, {err_msg}")
            return

        # Adjustments for distributed generation (outer loop)
        if ST.state.has_distributed_gen:
            # **PQV-mode DG adjustments (voltage-controlled generators)**
            if getattr(ST.state, "has_pqv_distributed_gen", False):
                wb = ST.state.working_buses        # DataFrame of current bus voltages
                pqv_df = ST.state.pqv_distributed_gen
                if pqv_df is not None and not pqv_df.empty:
                    # Get current voltages at PQV generator buses
                    volt_pqv = wb[wb['id'].isin(pqv_df['bus'])][['id', 'v_ph1', 'v_ph2', 'v_ph3']].copy()
                    for idx, dg in pqv_df.iterrows():
                        bus_id = dg['bus']
                        bus_volt = volt_pqv[volt_pqv['id'] == bus_id]
                        if bus_volt.empty:
                            continue  # skip if bus not found (should not happen if data is consistent)
                        # Old and new bus voltages (per-phase magnitudes)
                        old_volts = [dg['v_ph1'], dg['v_ph2'], dg['v_ph3']]
                        new_volts = [abs(bus_volt.iloc[0]['v_ph1']),
                                     abs(bus_volt.iloc[0]['v_ph2']),
                                     abs(bus_volt.iloc[0]['v_ph3'])]
                        # Generator parameters
                        v_set = dg['kv_set'] * 1000 / (3 ** 0.5)   # line-to-neutral setpoint voltage
                        xd = dg['xd']
                        p_phase = dg['kw_set'] * 1000 / 3         # per-phase active power (W)
                        w_ph1 = w_ph2 = w_ph3 = p_phase           # assume balanced P split
                        # Compute required reactive power per phase to reach v_set (approximate)
                        term1 = [(v_set * new_volts[i] / xd) ** 2 - p_phase ** 2 for i in range(3)]
                        # Avoid negative under sqrt (if voltage too low for given P)
                        term1 = [x if x > 0 else 0 for x in term1]
                        var_ph = [ (term1[i] ** 0.5) - ((new_volts[i] ** 2) / xd) for i in range(3) ]
                        # Clamp reactive power within DG limits (kvar_min <= Q <= kvar_max per phase)
                        qmin_per_phase = dg['kvar_min'] * 1000 / 3
                        qmax_per_phase = dg['kvar_max'] * 1000 / 3
                        var_ph[0] = max(var_ph[0], qmin_per_phase); var_ph[0] = min(var_ph[0], qmax_per_phase)
                        var_ph[1] = max(var_ph[1], qmin_per_phase); var_ph[1] = min(var_ph[1], qmax_per_phase)
                        var_ph[2] = max(var_ph[2], qmin_per_phase); var_ph[2] = min(var_ph[2], qmax_per_phase)
                        # Update loads: remove old DG load and add new one with updated P/Q
                        loads_df = ST.state.loads
                        dg_mode = str(dg['mode']).upper()
                        ST.state.loads = loads_df[~((loads_df['bus'] == bus_id) &
                                                    (loads_df['type'].str.upper() == dg_mode))].copy()
                        new_load = {
                            'bus': bus_id,
                            'conn': dg['conn'],
                            'type': dg['mode'],
                            'ph_1': -(w_ph1 + (var_ph[0] * 1j)),
                            'ph_2': -(w_ph2 + (var_ph[1] * 1j)),
                            'ph_3': -(w_ph3 + (var_ph[2] * 1j))
                        }
                        # Preserve any additional load columns (e.g., k_1, k_2, k_3 for other load types)
                        for col in ['k_1', 'k_2', 'k_3']:
                            if col in ST.state.loads.columns:
                                new_load[col] = None
                        ST.state.loads = pd.concat([ST.state.loads, pd.DataFrame([new_load])], ignore_index=True)
                        # Calculate max relative voltage change for this DG
                        volt_diffs = [abs((old_volts[i] - new_volts[i]) / new_volts[i]) if new_volts[i] != 0 else 0
                                      for i in range(3)]
                        max_volt_diff = max(volt_diffs) if volt_diffs else 0.0
                        # Update the PQV DG record for next iteration
                        ST.state.pqv_distributed_gen.at[idx, 'v_ph1'] = new_volts[0]
                        ST.state.pqv_distributed_gen.at[idx, 'v_ph2'] = new_volts[1]
                        ST.state.pqv_distributed_gen.at[idx, 'v_ph3'] = new_volts[2]
                        ST.state.pqv_distributed_gen.at[idx, 'max_diff'] = max_volt_diff
                        ST.state.pqv_distributed_gen.at[idx, 'w_ph1'] = w_ph1
                        ST.state.pqv_distributed_gen.at[idx, 'w_ph2'] = w_ph2
                        ST.state.pqv_distributed_gen.at[idx, 'w_ph3'] = w_ph3
                        ST.state.pqv_distributed_gen.at[idx, 'var_ph1'] = var_ph[0]
                        ST.state.pqv_distributed_gen.at[idx, 'var_ph2'] = var_ph[1]
                        ST.state.pqv_distributed_gen.at[idx, 'var_ph3'] = var_ph[2]
                        # Update generation register entry for this DG (remove old, add new)
                        gen_reg = ST.state.generation_register
                        ST.state.generation_register = gen_reg[gen_reg['bus'] != bus_id].copy()
                        ST.state.generation_register = pd.concat([ST.state.generation_register, pd.DataFrame([{
                            'bus': bus_id,
                            'mode': dg['mode'],
                            'conn': dg['conn'],
                            'kw_ph1': w_ph1 / 1000,   'kvar_ph1': var_ph[0] / 1000,
                            'kw_ph2': w_ph2 / 1000,   'kvar_ph2': var_ph[1] / 1000,
                            'kw_ph3': w_ph3 / 1000,   'kvar_ph3': var_ph[2] / 1000,
                            'max_diff': max_volt_diff}])], ignore_index=True)
            # **PI-mode DG adjustments (current-limited generators)**
            if getattr(ST.state, "has_pi_distributed_gen", False):
                wb = ST.state.working_buses
                pi_df = ST.state.pi_distributed_gen
                if pi_df is not None and not pi_df.empty:
                    volt_pi = wb[wb['id'].isin(pi_df['bus'])][['id', 'v_ph1', 'v_ph2', 'v_ph3']].copy()
                    for idx, dg in pi_df.iterrows():
                        bus_id = dg['bus']
                        bus_volt = volt_pi[volt_pi['id'] == bus_id]
                        if bus_volt.empty:
                            continue
                        # Old and new bus voltages
                        old_volts = [dg['v_ph1'], dg['v_ph2'], dg['v_ph3']]
                        new_volts = [abs(bus_volt.iloc[0]['v_ph1']),
                                     abs(bus_volt.iloc[0]['v_ph2']),
                                     abs(bus_volt.iloc[0]['v_ph3'])]
                        # Generator parameters
                        i_set = dg['amp_set']
                        p_phase = dg['kw_set'] * 1000 / 3     # per-phase active power (W)
                        q_min = dg['kvar_min'] * 1000 / 3
                        q_max = dg['kvar_max'] * 1000 / 3
                        # Determine per-phase voltages based on connection
                        conn_type = str(dg['conn']).upper()
                        if conn_type == "Y":
                            v_ph = new_volts[:]               # use line-to-neutral voltages
                            i_ph = i_set
                        elif conn_type == "D":
                            # line-to-line voltages for delta connection
                            v_ph = [
                                abs(bus_volt.iloc[0]['v_ph1'] - bus_volt.iloc[0]['v_ph2']),
                                abs(bus_volt.iloc[0]['v_ph2'] - bus_volt.iloc[0]['v_ph3']),
                                abs(bus_volt.iloc[0]['v_ph3'] - bus_volt.iloc[0]['v_ph1'])
                            ]
                            i_ph = i_set / (3 ** 0.5)
                        else:
                            v_ph = new_volts[:]
                            i_ph = i_set
                        # Compute reactive power per phase based on current limit (I*V and P)
                        q_ph = []
                        for i in range(3):
                            val = (i_ph * v_ph[i]) ** 2 - p_phase ** 2
                            q_val = (val ** 0.5) if val >= 0 else q_min  # if negative, use minimum reactive
                            q_val = min(q_val, q_max)                   # cap at max reactive
                            q_ph.append(q_val)
                        # Update loads (remove old entry, add new one)
                        loads_df = ST.state.loads
                        dg_mode = str(dg['mode']).upper()
                        ST.state.loads = loads_df[~((loads_df['bus'] == bus_id) &
                                                    (loads_df['type'].str.upper() == dg_mode))].copy()
                        new_load = {
                            'bus': bus_id,
                            'conn': dg['conn'],
                            'type': dg['mode'],
                            'ph_1': -(p_phase + q_ph[0] * 1j),
                            'ph_2': -(p_phase + q_ph[1] * 1j),
                            'ph_3': -(p_phase + q_ph[2] * 1j)
                        }
                        for col in ['k_1', 'k_2', 'k_3']:
                            if col in ST.state.loads.columns:
                                new_load[col] = None
                        ST.state.loads = pd.concat([ST.state.loads, pd.DataFrame([new_load])], ignore_index=True)
                        # Calculate max relative voltage change
                        volt_diffs = [abs((old_volts[i] - new_volts[i]) / new_volts[i]) if new_volts[i] != 0 else 0
                                      for i in range(3)]
                        max_volt_diff = max(volt_diffs) if volt_diffs else 0.0
                        # Update the PI DG record for next iteration
                        ST.state.pi_distributed_gen.at[idx, 'v_ph1'] = new_volts[0]
                        ST.state.pi_distributed_gen.at[idx, 'v_ph2'] = new_volts[1]
                        ST.state.pi_distributed_gen.at[idx, 'v_ph3'] = new_volts[2]
                        ST.state.pi_distributed_gen.at[idx, 'max_diff'] = max_volt_diff
                        ST.state.pi_distributed_gen.at[idx, 'w_ph1'] = p_phase
                        ST.state.pi_distributed_gen.at[idx, 'w_ph2'] = p_phase
                        ST.state.pi_distributed_gen.at[idx, 'w_ph3'] = p_phase
                        ST.state.pi_distributed_gen.at[idx, 'var_ph1'] = q_ph[0]
                        ST.state.pi_distributed_gen.at[idx, 'var_ph2'] = q_ph[1]
                        ST.state.pi_distributed_gen.at[idx, 'var_ph3'] = q_ph[2]
                        # Update generation register for this DG
                        gen_reg = ST.state.generation_register
                        ST.state.generation_register = gen_reg[gen_reg['bus'] != bus_id].copy()
                        ST.state.generation_register = pd.concat([
                            ST.state.generation_register,
                            pd.DataFrame([{
                                'bus': bus_id, 'mode': dg['mode'], 'conn': dg['conn'],
                                'kw_ph1': w_ph1 / 1000, 'kvar_ph1': var_ph[0] / 1000,
                                'kw_ph2': w_ph2 / 1000, 'kvar_ph2': var_ph[1] / 1000,
                                'kw_ph3': w_ph3 / 1000, 'kvar_ph3': var_ph[2] / 1000,
                                'max_diff': max_volt_diff
                            }])
                        ], ignore_index=True)
            # Compute the maximum voltage difference across all DGs for convergence check
            if not ST.state.generation_register.empty and 'max_diff' in ST.state.generation_register.columns:
                max_diff = ST.state.generation_register['max_diff'].max()
            else:
                max_diff = 0.0
        else:
            max_diff = 0.0  # no distributed generation, no outer iteration needed

        ST.state.outer_iteration += 1
    # End of while loop (convergence reached or no DG adjustments needed)

    # 4. Output results or final status
    if not err_msg:  # err_msg is empty string if no errors
        if verbose != 0:
            print(f"Execution finished, {ST.state.outer_iteration} outer iterations, "
                  f"{ST.state.inner_iteration} inner iterations (for latest outer round), {tolerance} tolerance")
        # Write result files and display summary if requested
        print_results.results(display_summary, timestamp)
    else:
        # In case an error message was set (should have already returned earlier)
        print(f"Execution aborted, {err_msg}")
        return