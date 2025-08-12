import os
import numpy as np
import pandas as pd
import shared_state as ST
from datetime import datetime


def results(display_summary: bool, timestamp: bool):
    """
    Generate power flow result reports (voltages, currents, power flows, losses, and generation)
    and save to CSV files. Optionally display a summary of results.
    """
    # Copy necessary data from shared state
    wb = ST.state.working_buses.copy()  # DataFrame of working buses (solved voltages, etc.)
    lines_df = ST.state.lines.copy()  # DataFrame of line segments (with currents from power flow)
    aux_buses = ST.state.auxiliar_buses.copy() if ST.state.auxiliar_buses is not None else None
    output_dir = ST.state.output_dir if hasattr(ST.state, 'output_dir') else ""

    if wb is None or wb.empty or lines_df is None or lines_df.empty:
        raise RuntimeError("Required data (working_buses or lines) is missing or empty.")

    # Determine phase configuration for each bus (if not already present)
    if 'phases' not in wb.columns:
        wb['phases'] = None
    for i in wb.index:
        if wb.at[i, 'number'] != 1:  # not the substation
            bus_id = wb.at[i, 'id']
            seg = lines_df[lines_df['bus2'] == bus_id]
            if not seg.empty:
                wb.at[i, 'phases'] = seg.iloc[0]['phases']
        else:
            wb.at[i, 'phases'] = "abc"  # substation bus assumed three-phase

    # If distributed loads exist, assign phases to auxiliary buses
    if ST.state.has_distributed_load and aux_buses is not None and not aux_buses.empty:
        phase_map = wb.set_index('id')['phases'].to_dict()
        aux_buses['phases'] = aux_buses['busx'].map(phase_map)
        ST.state.auxiliar_buses = aux_buses  # update in state (for consistency)

    # Prepare voltage magnitude and angle reports per phase
    volts = wb[['id', 'number']].copy()
    volts_phases = volts.copy()
    volts_pu = volts.copy()
    # Initialize columns for phase voltages (actual and per unit)
    for col in ['volt_A', 'deg_A', 'volt_B', 'deg_B', 'volt_C', 'deg_C']:
        volts_phases[col] = np.nan
        volts_pu[col] = np.nan
    # Fill voltage reports based on each bus's phase connection
    for i in wb.index:
        ph_config = wb.at[i, 'phases']
        # Retrieve complex phase voltages and base voltage
        v1 = wb.at[i, 'v_ph1'] if 'v_ph1' in wb.columns else 0 + 0j
        v2 = wb.at[i, 'v_ph2'] if 'v_ph2' in wb.columns else 0 + 0j
        v3 = wb.at[i, 'v_ph3'] if 'v_ph3' in wb.columns else 0 + 0j
        v_base = wb.at[i, 'v_base'] if 'v_base' in wb.columns else 1.0  # line-to-neutral base volts
        if ph_config == "a":
            volts_phases.at[i, 'volt_A'] = round(abs(v1), 1)
            volts_phases.at[i, 'deg_A'] = round(np.degrees(np.angle(v1)), 2)
            volts_pu.at[i, 'volt_A'] = round(abs(v1) / v_base, 5)
            volts_pu.at[i, 'deg_A'] = round(np.degrees(np.angle(v1)), 2)
        elif ph_config == "b":
            volts_phases.at[i, 'volt_B'] = round(abs(v2), 1)
            volts_phases.at[i, 'deg_B'] = round(np.degrees(np.angle(v2)), 2)
            volts_pu.at[i, 'volt_B'] = round(abs(v2) / v_base, 5)
            volts_pu.at[i, 'deg_B'] = round(np.degrees(np.angle(v2)), 2)
        elif ph_config == "c":
            volts_phases.at[i, 'volt_C'] = round(abs(v3), 1)
            volts_phases.at[i, 'deg_C'] = round(np.degrees(np.angle(v3)), 2)
            volts_pu.at[i, 'volt_C'] = round(abs(v3) / v_base, 5)
            volts_pu.at[i, 'deg_C'] = round(np.degrees(np.angle(v3)), 2)
        elif ph_config == "ab":
            volts_phases.at[i, 'volt_A'] = round(abs(v1), 1)
            volts_phases.at[i, 'volt_B'] = round(abs(v2), 1)
            volts_phases.at[i, 'deg_A'] = round(np.degrees(np.angle(v1)), 2)
            volts_phases.at[i, 'deg_B'] = round(np.degrees(np.angle(v2)), 2)
            volts_pu.at[i, 'volt_A'] = round(abs(v1) / v_base, 5)
            volts_pu.at[i, 'volt_B'] = round(abs(v2) / v_base, 5)
            volts_pu.at[i, 'deg_A'] = round(np.degrees(np.angle(v1)), 2)
            volts_pu.at[i, 'deg_B'] = round(np.degrees(np.angle(v2)), 2)
        elif ph_config == "bc":
            volts_phases.at[i, 'volt_B'] = round(abs(v2), 1)
            volts_phases.at[i, 'volt_C'] = round(abs(v3), 1)
            volts_phases.at[i, 'deg_B'] = round(np.degrees(np.angle(v2)), 2)
            volts_phases.at[i, 'deg_C'] = round(np.degrees(np.angle(v3)), 2)
            volts_pu.at[i, 'volt_B'] = round(abs(v2) / v_base, 5)
            volts_pu.at[i, 'volt_C'] = round(abs(v3) / v_base, 5)
            volts_pu.at[i, 'deg_B'] = round(np.degrees(np.angle(v2)), 2)
            volts_pu.at[i, 'deg_C'] = round(np.degrees(np.angle(v3)), 2)
        elif ph_config == "ac":
            volts_phases.at[i, 'volt_A'] = round(abs(v1), 1)
            volts_phases.at[i, 'volt_C'] = round(abs(v3), 1)
            volts_phases.at[i, 'deg_A'] = round(np.degrees(np.angle(v1)), 2)
            volts_phases.at[i, 'deg_C'] = round(np.degrees(np.angle(v3)), 2)
            volts_pu.at[i, 'volt_A'] = round(abs(v1) / v_base, 5)
            volts_pu.at[i, 'volt_C'] = round(abs(v3) / v_base, 5)
            volts_pu.at[i, 'deg_A'] = round(np.degrees(np.angle(v1)), 2)
            volts_pu.at[i, 'deg_C'] = round(np.degrees(np.angle(v3)), 2)
        elif ph_config == "abc":
            volts_phases.at[i, 'volt_A'] = round(abs(v1), 1)
            volts_phases.at[i, 'volt_B'] = round(abs(v2), 1)
            volts_phases.at[i, 'volt_C'] = round(abs(v3), 1)
            volts_phases.at[i, 'deg_A'] = round(np.degrees(np.angle(v1)), 2)
            volts_phases.at[i, 'deg_B'] = round(np.degrees(np.angle(v2)), 2)
            volts_phases.at[i, 'deg_C'] = round(np.degrees(np.angle(v3)), 2)
            volts_pu.at[i, 'volt_A'] = round(abs(v1) / v_base, 5)
            volts_pu.at[i, 'volt_B'] = round(abs(v2) / v_base, 5)
            volts_pu.at[i, 'volt_C'] = round(abs(v3) / v_base, 5)
            volts_pu.at[i, 'deg_A'] = round(np.degrees(np.angle(v1)), 2)
            volts_pu.at[i, 'deg_B'] = round(np.degrees(np.angle(v2)), 2)
            volts_pu.at[i, 'deg_C'] = round(np.degrees(np.angle(v3)), 2)
    # Drop 'number' column and sort by bus ID
    volts_phases.drop(columns=['number'], inplace=True)
    volts_phases.sort_values(by='id', inplace=True)
    volts_pu.drop(columns=['number'], inplace=True)
    volts_pu.sort_values(by='id', inplace=True)

    # Identify maximum and minimum per-unit voltages across all buses
    ext_v_pu = pd.DataFrame([[0.0, 0, 2.0, 0]], columns=['max', 'bus_max', 'min', 'bus_min'])
    for i in volts_pu.index:
        for phase_col in ['volt_A', 'volt_B', 'volt_C']:
            val = volts_pu.at[i, phase_col]
            if pd.notna(val):
                if val > ext_v_pu.at[0, 'max']:
                    ext_v_pu.at[0, 'max'] = val
                    ext_v_pu.at[0, 'bus_max'] = volts_pu.at[i, 'id']
                if val < ext_v_pu.at[0, 'min']:
                    ext_v_pu.at[0, 'min'] = val
                    ext_v_pu.at[0, 'bus_min'] = volts_pu.at[i, 'id']

    # Compute line-to-line voltages at each bus (for buses with multi-phase connections)
    volts_lines = volts.copy()
    for col in ['volt_AB', 'deg_AB', 'volt_BC', 'deg_BC', 'volt_CA', 'deg_CA']:
        volts_lines[col] = np.nan
    for i in wb.index:
        ph_config = wb.at[i, 'phases']
        v1 = wb.at[i, 'v_ph1'] if 'v_ph1' in wb.columns else 0 + 0j
        v2 = wb.at[i, 'v_ph2'] if 'v_ph2' in wb.columns else 0 + 0j
        v3 = wb.at[i, 'v_ph3'] if 'v_ph3' in wb.columns else 0 + 0j
        if ph_config == "ab":
            volts_lines.at[i, 'volt_AB'] = round(abs(v1 - v2), 1)
            volts_lines.at[i, 'deg_AB'] = round(np.degrees(np.angle(v1 - v2)), 2)
        elif ph_config == "bc":
            volts_lines.at[i, 'volt_BC'] = round(abs(v2 - v3), 1)
            volts_lines.at[i, 'deg_BC'] = round(np.degrees(np.angle(v2 - v3)), 2)
        elif ph_config == "ac":
            volts_lines.at[i, 'volt_CA'] = round(abs(v3 - v1), 1)
            volts_lines.at[i, 'deg_CA'] = round(np.degrees(np.angle(v3 - v1)), 2)
        elif ph_config == "abc":
            volts_lines.at[i, 'volt_AB'] = round(abs(v1 - v2), 1)
            volts_lines.at[i, 'volt_BC'] = round(abs(v2 - v3), 1)
            volts_lines.at[i, 'volt_CA'] = round(abs(v3 - v1), 1)
            volts_lines.at[i, 'deg_AB'] = round(np.degrees(np.angle(v1 - v2)), 2)
            volts_lines.at[i, 'deg_BC'] = round(np.degrees(np.angle(v2 - v3)), 2)
            volts_lines.at[i, 'deg_CA'] = round(np.degrees(np.angle(v3 - v1)), 2)
    volts_lines.sort_values(by='number', inplace=True)
    volts_lines.drop(columns=['number'], inplace=True)

    # Construct lineflow DataFrame containing line currents and power flows
    lineflow = lines_df[['bus1', 'bus2', 'phases', 'ibus1_1', 'ibus1_2', 'ibus1_3']].copy()
    lineflow.rename(columns={'bus1': 'from', 'bus2': 'to',
                             'ibus1_1': 'in_I_ph1', 'ibus1_2': 'in_I_ph2', 'ibus1_3': 'in_I_ph3'}, inplace=True)
    # Merge receiving-end currents from working_buses (to get out_I at the 'to' side of each line)
    out_currents = wb[['id', 'ibus_1', 'ibus_2', 'ibus_3']].rename(columns={
        'id': 'to', 'ibus_1': 'out_I_ph1', 'ibus_2': 'out_I_ph2', 'ibus_3': 'out_I_ph3'})
    lineflow = lineflow.merge(out_currents, on='to', how='left')
    # Merge sending-end (from bus) and receiving-end (to bus) voltages for power flow calculation
    from_voltages = wb[['id', 'v_ph1', 'v_ph2', 'v_ph3']].rename(columns={
        'id': 'from', 'v_ph1': 'v1_from', 'v_ph2': 'v2_from', 'v_ph3': 'v3_from'})
    to_voltages = wb[['id', 'v_ph1', 'v_ph2', 'v_ph3']].rename(columns={
        'id': 'to', 'v_ph1': 'v1_to', 'v_ph2': 'v2_to', 'v_ph3': 'v3_to'})
    lineflow = lineflow.merge(from_voltages, on='from', how='left')
    lineflow = lineflow.merge(to_voltages, on='to', how='left')
    # Compute complex power at sending and receiving ends of each line (S = V * I*)
    lineflow['in_S_ph1'] = lineflow['v1_from'] * np.conjugate(lineflow['in_I_ph1'])
    lineflow['in_S_ph2'] = lineflow['v2_from'] * np.conjugate(lineflow['in_I_ph2'])
    lineflow['in_S_ph3'] = lineflow['v3_from'] * np.conjugate(lineflow['in_I_ph3'])
    lineflow['out_S_ph1'] = lineflow['v1_to'] * np.conjugate(lineflow['out_I_ph1'])
    lineflow['out_S_ph2'] = lineflow['v2_to'] * np.conjugate(lineflow['out_I_ph2'])
    lineflow['out_S_ph3'] = lineflow['v3_to'] * np.conjugate(lineflow['out_I_ph3'])
    # Calculate line losses per phase (difference between sending and receiving end power) and total losses
    lineflow['ploss_ph1'] = np.real(lineflow['in_S_ph1']) - np.real(lineflow['out_S_ph1'])
    lineflow['ploss_ph2'] = np.real(lineflow['in_S_ph2']) - np.real(lineflow['out_S_ph2'])
    lineflow['ploss_ph3'] = np.real(lineflow['in_S_ph3']) - np.real(lineflow['out_S_ph3'])
    lineflow['ploss_totals'] = (lineflow['ploss_ph1'] + lineflow['ploss_ph2'] + lineflow['ploss_ph3']).round(1)
    lineflow['qloss_ph1'] = np.imag(lineflow['in_S_ph1']) - np.imag(lineflow['out_S_ph1'])
    lineflow['qloss_ph2'] = np.imag(lineflow['in_S_ph2']) - np.imag(lineflow['out_S_ph2'])
    lineflow['qloss_ph3'] = np.imag(lineflow['in_S_ph3']) - np.imag(lineflow['out_S_ph3'])
    lineflow['qloss_totals'] = (lineflow['qloss_ph1'] + lineflow['qloss_ph2'] + lineflow['qloss_ph3']).round(1)
    lineflow.sort_values(by=['from', 'to'], inplace=True)

    # Prepare current flow report (magnitudes and angles of line currents)
    cflow = lineflow[['from', 'to', 'phases']].copy()
    for col in ['amp_in_I_ph1', 'deg_in_I_ph1', 'amp_in_I_ph2', 'deg_in_I_ph2',
                'amp_in_I_ph3', 'deg_in_I_ph3', 'amp_out_I_ph1', 'deg_out_I_ph1',
                'amp_out_I_ph2', 'deg_out_I_ph2', 'amp_out_I_ph3', 'deg_out_I_ph3']:
        cflow[col] = np.nan
    for idx in lineflow.index:
        ph = lineflow.at[idx, 'phases']
        # Compute magnitude (abs) and angle (deg) of currents, per present phase(s)
        if ph == "a":
            cflow.at[idx, 'amp_in_I_ph1'] = round(abs(lineflow.at[idx, 'in_I_ph1']), 2)
            cflow.at[idx, 'deg_in_I_ph1'] = round(np.degrees(np.angle(lineflow.at[idx, 'in_I_ph1'])), 2)
            cflow.at[idx, 'amp_out_I_ph1'] = round(abs(lineflow.at[idx, 'out_I_ph1']), 2)
            cflow.at[idx, 'deg_out_I_ph1'] = round(np.degrees(np.angle(lineflow.at[idx, 'out_I_ph1'])), 2)
        elif ph == "b":
            cflow.at[idx, 'amp_in_I_ph2'] = round(abs(lineflow.at[idx, 'in_I_ph2']), 2)
            cflow.at[idx, 'deg_in_I_ph2'] = round(np.degrees(np.angle(lineflow.at[idx, 'in_I_ph2'])), 2)
            cflow.at[idx, 'amp_out_I_ph2'] = round(abs(lineflow.at[idx, 'out_I_ph2']), 2)
            cflow.at[idx, 'deg_out_I_ph2'] = round(np.degrees(np.angle(lineflow.at[idx, 'out_I_ph2'])), 2)
        elif ph == "c":
            cflow.at[idx, 'amp_in_I_ph3'] = round(abs(lineflow.at[idx, 'in_I_ph3']), 2)
            cflow.at[idx, 'deg_in_I_ph3'] = round(np.degrees(np.angle(lineflow.at[idx, 'in_I_ph3'])), 2)
            cflow.at[idx, 'amp_out_I_ph3'] = round(abs(lineflow.at[idx, 'out_I_ph3']), 2)
            cflow.at[idx, 'deg_out_I_ph3'] = round(np.degrees(np.angle(lineflow.at[idx, 'out_I_ph3'])), 2)
        elif ph == "ab":
            cflow.at[idx, 'amp_in_I_ph1'] = round(abs(lineflow.at[idx, 'in_I_ph1']), 2)
            cflow.at[idx, 'deg_in_I_ph1'] = round(np.degrees(np.angle(lineflow.at[idx, 'in_I_ph1'])), 2)
            cflow.at[idx, 'amp_out_I_ph1'] = round(abs(lineflow.at[idx, 'out_I_ph1']), 2)
            cflow.at[idx, 'deg_out_I_ph1'] = round(np.degrees(np.angle(lineflow.at[idx, 'out_I_ph1'])), 2)
            cflow.at[idx, 'amp_in_I_ph2'] = round(abs(lineflow.at[idx, 'in_I_ph2']), 2)
            cflow.at[idx, 'deg_in_I_ph2'] = round(np.degrees(np.angle(lineflow.at[idx, 'in_I_ph2'])), 2)
            cflow.at[idx, 'amp_out_I_ph2'] = round(abs(lineflow.at[idx, 'out_I_ph2']), 2)
            cflow.at[idx, 'deg_out_I_ph2'] = round(np.degrees(np.angle(lineflow.at[idx, 'out_I_ph2'])), 2)
        elif ph == "ac":
            cflow.at[idx, 'amp_in_I_ph1'] = round(abs(lineflow.at[idx, 'in_I_ph1']), 2)
            cflow.at[idx, 'deg_in_I_ph1'] = round(np.degrees(np.angle(lineflow.at[idx, 'in_I_ph1'])), 2)
            cflow.at[idx, 'amp_out_I_ph1'] = round(abs(lineflow.at[idx, 'out_I_ph1']), 2)
            cflow.at[idx, 'deg_out_I_ph1'] = round(np.degrees(np.angle(lineflow.at[idx, 'out_I_ph1'])), 2)
            cflow.at[idx, 'amp_in_I_ph3'] = round(abs(lineflow.at[idx, 'in_I_ph3']), 2)
            cflow.at[idx, 'deg_in_I_ph3'] = round(np.degrees(np.angle(lineflow.at[idx, 'in_I_ph3'])), 2)
            cflow.at[idx, 'amp_out_I_ph3'] = round(abs(lineflow.at[idx, 'out_I_ph3']), 2)
            cflow.at[idx, 'deg_out_I_ph3'] = round(np.degrees(np.angle(lineflow.at[idx, 'out_I_ph3'])), 2)
        elif ph == "bc":
            cflow.at[idx, 'amp_in_I_ph2'] = round(abs(lineflow.at[idx, 'in_I_ph2']), 2)
            cflow.at[idx, 'deg_in_I_ph2'] = round(np.degrees(np.angle(lineflow.at[idx, 'in_I_ph2'])), 2)
            cflow.at[idx, 'amp_out_I_ph2'] = round(abs(lineflow.at[idx, 'out_I_ph2']), 2)
            cflow.at[idx, 'deg_out_I_ph2'] = round(np.degrees(np.angle(lineflow.at[idx, 'out_I_ph2'])), 2)
            cflow.at[idx, 'amp_in_I_ph3'] = round(abs(lineflow.at[idx, 'in_I_ph3']), 2)
            cflow.at[idx, 'deg_in_I_ph3'] = round(np.degrees(np.angle(lineflow.at[idx, 'in_I_ph3'])), 2)
            cflow.at[idx, 'amp_out_I_ph3'] = round(abs(lineflow.at[idx, 'out_I_ph3']), 2)
            cflow.at[idx, 'deg_out_I_ph3'] = round(np.degrees(np.angle(lineflow.at[idx, 'out_I_ph3'])), 2)
        elif ph == "abc":
            cflow.at[idx, 'amp_in_I_ph1'] = round(abs(lineflow.at[idx, 'in_I_ph1']), 2)
            cflow.at[idx, 'deg_in_I_ph1'] = round(np.degrees(np.angle(lineflow.at[idx, 'in_I_ph1'])), 2)
            cflow.at[idx, 'amp_out_I_ph1'] = round(abs(lineflow.at[idx, 'out_I_ph1']), 2)
            cflow.at[idx, 'deg_out_I_ph1'] = round(np.degrees(np.angle(lineflow.at[idx, 'out_I_ph1'])), 2)
            cflow.at[idx, 'amp_in_I_ph2'] = round(abs(lineflow.at[idx, 'in_I_ph2']), 2)
            cflow.at[idx, 'deg_in_I_ph2'] = round(np.degrees(np.angle(lineflow.at[idx, 'in_I_ph2'])), 2)
            cflow.at[idx, 'amp_out_I_ph2'] = round(abs(lineflow.at[idx, 'out_I_ph2']), 2)
            cflow.at[idx, 'deg_out_I_ph2'] = round(np.degrees(np.angle(lineflow.at[idx, 'out_I_ph2'])), 2)
            cflow.at[idx, 'amp_in_I_ph3'] = round(abs(lineflow.at[idx, 'in_I_ph3']), 2)
            cflow.at[idx, 'deg_in_I_ph3'] = round(np.degrees(np.angle(lineflow.at[idx, 'in_I_ph3'])), 2)
            cflow.at[idx, 'amp_out_I_ph3'] = round(abs(lineflow.at[idx, 'out_I_ph3']), 2)
            cflow.at[idx, 'deg_out_I_ph3'] = round(np.degrees(np.angle(lineflow.at[idx, 'out_I_ph3'])), 2)

    # Prepare power flow report (kW/kVAr in and out for each line)
    pflow = lineflow[['from', 'to', 'phases']].copy()
    for col in ['kW_in_ph1', 'kVAr_in_ph1', 'kW_in_ph2', 'kVAr_in_ph2', 'kW_in_ph3', 'kVAr_in_ph3',
                'kW_out_ph1', 'kVAr_out_ph1', 'kW_out_ph2', 'kVAr_out_ph2', 'kW_out_ph3', 'kVAr_out_ph3']:
        pflow[col] = np.nan
    for idx in lineflow.index:
        ph = lineflow.at[idx, 'phases']
        # Use real and imaginary parts of complex power to compute kW and kVAr (divide by 1000 for kW/kVAr)
        if ph == "a":
            pflow.at[idx, 'kW_in_ph1'] = round(lineflow.at[idx, 'in_S_ph1'].real / 1000, 3)
            pflow.at[idx, 'kW_out_ph1'] = round(lineflow.at[idx, 'out_S_ph1'].real / 1000, 3)
            pflow.at[idx, 'kVAr_in_ph1'] = round(lineflow.at[idx, 'in_S_ph1'].imag / 1000, 3)
            pflow.at[idx, 'kVAr_out_ph1'] = round(lineflow.at[idx, 'out_S_ph1'].imag / 1000, 3)
        elif ph == "b":
            pflow.at[idx, 'kW_in_ph2'] = round(lineflow.at[idx, 'in_S_ph2'].real / 1000, 3)
            pflow.at[idx, 'kW_out_ph2'] = round(lineflow.at[idx, 'out_S_ph2'].real / 1000, 3)
            pflow.at[idx, 'kVAr_in_ph2'] = round(lineflow.at[idx, 'in_S_ph2'].imag / 1000, 3)
            pflow.at[idx, 'kVAr_out_ph2'] = round(lineflow.at[idx, 'out_S_ph2'].imag / 1000, 3)
        elif ph == "c":
            pflow.at[idx, 'kW_in_ph3'] = round(lineflow.at[idx, 'in_S_ph3'].real / 1000, 3)
            pflow.at[idx, 'kW_out_ph3'] = round(lineflow.at[idx, 'out_S_ph3'].real / 1000, 3)
            pflow.at[idx, 'kVAr_in_ph3'] = round(lineflow.at[idx, 'in_S_ph3'].imag / 1000, 3)
            pflow.at[idx, 'kVAr_out_ph3'] = round(lineflow.at[idx, 'out_S_ph3'].imag / 1000, 3)
        elif ph == "ab":
            pflow.at[idx, 'kW_in_ph1'] = round(lineflow.at[idx, 'in_S_ph1'].real / 1000, 3)
            pflow.at[idx, 'kW_out_ph1'] = round(lineflow.at[idx, 'out_S_ph1'].real / 1000, 3)
            pflow.at[idx, 'kVAr_in_ph1'] = round(lineflow.at[idx, 'in_S_ph1'].imag / 1000, 3)
            pflow.at[idx, 'kVAr_out_ph1'] = round(lineflow.at[idx, 'out_S_ph1'].imag / 1000, 3)
            pflow.at[idx, 'kW_in_ph2'] = round(lineflow.at[idx, 'in_S_ph2'].real / 1000, 3)
            pflow.at[idx, 'kW_out_ph2'] = round(lineflow.at[idx, 'out_S_ph2'].real / 1000, 3)
            pflow.at[idx, 'kVAr_in_ph2'] = round(lineflow.at[idx, 'in_S_ph2'].imag / 1000, 3)
            pflow.at[idx, 'kVAr_out_ph2'] = round(lineflow.at[idx, 'out_S_ph2'].imag / 1000, 3)
        elif ph == "ac":
            pflow.at[idx, 'kW_in_ph1'] = round(lineflow.at[idx, 'in_S_ph1'].real / 1000, 3)
            pflow.at[idx, 'kW_out_ph1'] = round(lineflow.at[idx, 'out_S_ph1'].real / 1000, 3)
            pflow.at[idx, 'kVAr_in_ph1'] = round(lineflow.at[idx, 'in_S_ph1'].imag / 1000, 3)
            pflow.at[idx, 'kVAr_out_ph1'] = round(lineflow.at[idx, 'out_S_ph1'].imag / 1000, 3)
            pflow.at[idx, 'kW_in_ph3'] = round(lineflow.at[idx, 'in_S_ph3'].real / 1000, 3)
            pflow.at[idx, 'kW_out_ph3'] = round(lineflow.at[idx, 'out_S_ph3'].real / 1000, 3)
            pflow.at[idx, 'kVAr_in_ph3'] = round(lineflow.at[idx, 'in_S_ph3'].imag / 1000, 3)
            pflow.at[idx, 'kVAr_out_ph3'] = round(lineflow.at[idx, 'out_S_ph3'].imag / 1000, 3)
        elif ph == "bc":
            pflow.at[idx, 'kW_in_ph2'] = round(lineflow.at[idx, 'in_S_ph2'].real / 1000, 3)
            pflow.at[idx, 'kW_out_ph2'] = round(lineflow.at[idx, 'out_S_ph2'].real / 1000, 3)
            pflow.at[idx, 'kVAr_in_ph2'] = round(lineflow.at[idx, 'in_S_ph2'].imag / 1000, 3)
            pflow.at[idx, 'kVAr_out_ph2'] = round(lineflow.at[idx, 'out_S_ph2'].imag / 1000, 3)
            pflow.at[idx, 'kW_in_ph3'] = round(lineflow.at[idx, 'in_S_ph3'].real / 1000, 3)
            pflow.at[idx, 'kW_out_ph3'] = round(lineflow.at[idx, 'out_S_ph3'].real / 1000, 3)
            pflow.at[idx, 'kVAr_in_ph3'] = round(lineflow.at[idx, 'in_S_ph3'].imag / 1000, 3)
            pflow.at[idx, 'kVAr_out_ph3'] = round(lineflow.at[idx, 'out_S_ph3'].imag / 1000, 3)
        elif ph == "abc":
            pflow.at[idx, 'kW_in_ph1'] = round(lineflow.at[idx, 'in_S_ph1'].real / 1000, 3)
            pflow.at[idx, 'kW_out_ph1'] = round(lineflow.at[idx, 'out_S_ph1'].real / 1000, 3)
            pflow.at[idx, 'kVAr_in_ph1'] = round(lineflow.at[idx, 'in_S_ph1'].imag / 1000, 3)
            pflow.at[idx, 'kVAr_out_ph1'] = round(lineflow.at[idx, 'out_S_ph1'].imag / 1000, 3)
            pflow.at[idx, 'kW_in_ph2'] = round(lineflow.at[idx, 'in_S_ph2'].real / 1000, 3)
            pflow.at[idx, 'kW_out_ph2'] = round(lineflow.at[idx, 'out_S_ph2'].real / 1000, 3)
            pflow.at[idx, 'kVAr_in_ph2'] = round(lineflow.at[idx, 'in_S_ph2'].imag / 1000, 3)
            pflow.at[idx, 'kVAr_out_ph2'] = round(lineflow.at[idx, 'out_S_ph2'].imag / 1000, 3)
            pflow.at[idx, 'kW_in_ph3'] = round(lineflow.at[idx, 'in_S_ph3'].real / 1000, 3)
            pflow.at[idx, 'kW_out_ph3'] = round(lineflow.at[idx, 'out_S_ph3'].real / 1000, 3)
            pflow.at[idx, 'kVAr_in_ph3'] = round(lineflow.at[idx, 'in_S_ph3'].imag / 1000, 3)
            pflow.at[idx, 'kVAr_out_ph3'] = round(lineflow.at[idx, 'out_S_ph3'].imag / 1000, 3)

    # Prepare losses report per line
    losses = lineflow[['from', 'to', 'ploss_totals', 'qloss_totals']].copy()
    for col in ['ploss_ph1', 'qloss_ph1', 'ploss_ph2', 'qloss_ph2', 'ploss_ph3', 'qloss_ph3']:
        losses[col] = np.nan
    for idx in lineflow.index:
        ph = lineflow.at[idx, 'phases']
        if ph == "a":
            losses.at[idx, 'ploss_ph1'] = round(lineflow.at[idx, 'ploss_ph1'], 1)
            losses.at[idx, 'qloss_ph1'] = round(lineflow.at[idx, 'qloss_ph1'], 1)
        elif ph == "b":
            losses.at[idx, 'ploss_ph2'] = round(lineflow.at[idx, 'ploss_ph2'], 1)
            losses.at[idx, 'qloss_ph2'] = round(lineflow.at[idx, 'qloss_ph2'], 1)
        elif ph == "c":
            losses.at[idx, 'ploss_ph3'] = round(lineflow.at[idx, 'ploss_ph3'], 1)
            losses.at[idx, 'qloss_ph3'] = round(lineflow.at[idx, 'qloss_ph3'], 1)
        elif ph == "ab":
            losses.at[idx, 'ploss_ph1'] = round(lineflow.at[idx, 'ploss_ph1'], 1)
            losses.at[idx, 'qloss_ph1'] = round(lineflow.at[idx, 'qloss_ph1'], 1)
            losses.at[idx, 'ploss_ph2'] = round(lineflow.at[idx, 'ploss_ph2'], 1)
            losses.at[idx, 'qloss_ph2'] = round(lineflow.at[idx, 'qloss_ph2'], 1)
        elif ph == "ac":
            losses.at[idx, 'ploss_ph1'] = round(lineflow.at[idx, 'ploss_ph1'], 1)
            losses.at[idx, 'qloss_ph1'] = round(lineflow.at[idx, 'qloss_ph1'], 1)
            losses.at[idx, 'ploss_ph3'] = round(lineflow.at[idx, 'ploss_ph3'], 1)
            losses.at[idx, 'qloss_ph3'] = round(lineflow.at[idx, 'qloss_ph3'], 1)
        elif ph == "bc":
            losses.at[idx, 'ploss_ph2'] = round(lineflow.at[idx, 'ploss_ph2'], 1)
            losses.at[idx, 'qloss_ph2'] = round(lineflow.at[idx, 'qloss_ph2'], 1)
            losses.at[idx, 'ploss_ph3'] = round(lineflow.at[idx, 'ploss_ph3'], 1)
            losses.at[idx, 'qloss_ph3'] = round(lineflow.at[idx, 'qloss_ph3'], 1)
        elif ph == "abc":
            losses.at[idx, 'ploss_ph1'] = round(lineflow.at[idx, 'ploss_ph1'], 1)
            losses.at[idx, 'qloss_ph1'] = round(lineflow.at[idx, 'qloss_ph1'], 1)
            losses.at[idx, 'ploss_ph2'] = round(lineflow.at[idx, 'ploss_ph2'], 1)
            losses.at[idx, 'qloss_ph2'] = round(lineflow.at[idx, 'qloss_ph2'], 1)
            losses.at[idx, 'ploss_ph3'] = round(lineflow.at[idx, 'ploss_ph3'], 1)
            losses.at[idx, 'qloss_ph3'] = round(lineflow.at[idx, 'qloss_ph3'], 1)

    # If distributed loads present, combine any split segments (auxiliary buses) back into single entries
    if ST.state.has_distributed_load and aux_buses is not None and not aux_buses.empty:
        aux_ids = aux_buses['busx'].tolist()
        # Remove auxiliary buses from voltage reports
        volts_phases = volts_phases[~volts_phases['id'].isin(aux_ids)]
        volts_pu = volts_pu[~volts_pu['id'].isin(aux_ids)]
        volts_lines = volts_lines[~volts_lines['id'].isin(aux_ids)]
        volts_lines.sort_values(by='id', inplace=True)
        # Combine current flows for segments split by an auxiliary bus
        cflow_aux_in = cflow[cflow['to'].isin(aux_ids)].copy().rename(columns={'to': 'busx'})
        cflow_aux_out = cflow[cflow['from'].isin(aux_ids)].copy().rename(columns={'from': 'busx'})
        cflow_aux = pd.merge(cflow_aux_in, cflow_aux_out, on='busx', suffixes=('_in', '_out'))
        cflow_aux = cflow_aux.rename(columns={'from_in': 'from', 'to_out': 'to'}).drop(
            columns=['busx', 'to_in', 'from_out', 'phases_in', 'phases_out'], errors='ignore')
        cflow_aux = cflow_aux[cflow.columns]  # align columns
        cflow = cflow[~(cflow['to'].isin(aux_ids) | cflow['from'].isin(aux_ids))]
        cflow = pd.concat([cflow, cflow_aux], ignore_index=True)
        cflow.sort_values(by=['from', 'to'], inplace=True)
        # Combine power flows for segments split by an auxiliary bus
        pflow_aux_in = pflow[pflow['to'].isin(aux_ids)].copy().rename(columns={'to': 'busx'})
        pflow_aux_out = pflow[pflow['from'].isin(aux_ids)].copy().rename(columns={'from': 'busx'})
        pflow_aux = pd.merge(pflow_aux_in, pflow_aux_out, on='busx', suffixes=('_in', '_out'))
        pflow_aux = pflow_aux.rename(columns={'from_in': 'from', 'to_out': 'to'}).drop(
            columns=['busx', 'to_in', 'from_out', 'phases_in', 'phases_out'], errors='ignore')
        pflow_aux = pflow_aux[pflow.columns]
        pflow = pflow[~(pflow['to'].isin(aux_ids) | pflow['from'].isin(aux_ids))]
        pflow = pd.concat([pflow, pflow_aux], ignore_index=True)
        pflow.sort_values(by=['from', 'to'], inplace=True)
        # Combine loss entries for split segments
        aux_losses_1 = losses[(losses['from'].isin(aux_ids)) | (losses['to'].isin(aux_ids))].copy()
        losses = losses[~((losses['from'].isin(aux_ids)) | (losses['to'].isin(aux_ids)))]
        aux_losses_combined = []
        for i in aux_losses_1.index:
            for j in aux_losses_1.index:
                if aux_losses_1.at[i, 'to'] == aux_losses_1.at[j, 'from']:
                    busx_id = aux_losses_1.at[i, 'to']
                    phases = aux_buses[aux_buses['busx'] == busx_id]['phases']
                    phases = phases.iloc[0] if not phases.empty else ""
                    orig_from = aux_losses_1.at[i, 'from']
                    orig_to = aux_losses_1.at[j, 'to']
                    # Sum the two segments' losses for each phase
                    ploss_tot = aux_losses_1.at[i, 'ploss_totals'] + aux_losses_1.at[j, 'ploss_totals']
                    qloss_tot = aux_losses_1.at[i, 'qloss_totals'] + aux_losses_1.at[j, 'qloss_totals']
                    ploss_ph1 = aux_losses_1.at[i, 'ploss_ph1'] + aux_losses_1.at[
                        j, 'ploss_ph1'] if 'a' in phases else 0.0
                    qloss_ph1 = aux_losses_1.at[i, 'qloss_ph1'] + aux_losses_1.at[
                        j, 'qloss_ph1'] if 'a' in phases else 0.0
                    ploss_ph2 = aux_losses_1.at[i, 'ploss_ph2'] + aux_losses_1.at[
                        j, 'ploss_ph2'] if 'b' in phases else 0.0
                    qloss_ph2 = aux_losses_1.at[i, 'qloss_ph2'] + aux_losses_1.at[
                        j, 'qloss_ph2'] if 'b' in phases else 0.0
                    ploss_ph3 = aux_losses_1.at[i, 'ploss_ph3'] + aux_losses_1.at[
                        j, 'ploss_ph3'] if 'c' in phases else 0.0
                    qloss_ph3 = aux_losses_1.at[i, 'qloss_ph3'] + aux_losses_1.at[
                        j, 'qloss_ph3'] if 'c' in phases else 0.0
                    aux_losses_combined.append({
                        'from': orig_from, 'to': orig_to,
                        'ploss_ph1': round(ploss_ph1, 1), 'qloss_ph1': round(qloss_ph1, 1),
                        'ploss_ph2': round(ploss_ph2, 1), 'qloss_ph2': round(qloss_ph2, 1),
                        'ploss_ph3': round(ploss_ph3, 1), 'qloss_ph3': round(qloss_ph3, 1),
                        'ploss_totals': round(ploss_tot, 1), 'qloss_totals': round(qloss_tot, 1)
                    })
        if aux_losses_combined:
            losses = pd.concat([losses, pd.DataFrame(aux_losses_combined)], ignore_index=True)
        losses.sort_values(by=['from', 'to'], inplace=True)

    # Prepare distributed generation register for output, if applicable
    gen_reg = None
    if ST.state.has_distributed_gen:
        gen_reg = ST.state.generation_register.copy()
        if 'max_diff' in gen_reg.columns:
            gen_reg.drop(columns=['max_diff'], inplace=True)
        for col in ['kw_ph1', 'kw_ph2', 'kw_ph3', 'kvar_ph1', 'kvar_ph2', 'kvar_ph3']:
            if col in gen_reg.columns:
                gen_reg[col] = gen_reg[col].round(3)

    # Ensure output directory exists
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    def write_csv(df, base_name):
        fname = f"{base_name}{('-' + datetime.now().strftime('%Y%m%d-%H%M')) if timestamp else ''}.csv"
        path = os.path.join(output_dir, fname) if output_dir else fname
        df.to_csv(path, index=False)

    # Write result CSV files
    write_csv(volts_phases, "sdpf_volts_phase")
    write_csv(volts_pu, "sdpf_volts_pu")
    write_csv(volts_lines, "sdpf_volts_line")
    write_csv(cflow.drop(columns=['phases']), "sdpf_current_flow")
    write_csv(pflow.drop(columns=['phases']), "sdpf_power_flow")
    # Total input power (sum of substation incoming power on each phase)
    sub_bus_id = int(ST.state.substation.iloc[0]['bus']) if ST.state.substation is not None else None
    if sub_bus_id is not None:
        mask = (pflow['from'] == sub_bus_id)
    else:
        mask = pd.Series([True] * len(pflow))
    tip_df = pflow[mask][['kW_in_ph1', 'kW_in_ph2', 'kW_in_ph3', 'kVAr_in_ph1', 'kVAr_in_ph2', 'kVAr_in_ph3']].copy()
    if tip_df.empty:
        total_input_power = pd.DataFrame([[0, 0, 0, 0, 0, 0]],
                                         columns=['kW_in_ph1', 'kW_in_ph2', 'kW_in_ph3', 'kVAr_in_ph1', 'kVAr_in_ph2',
                                                  'kVAr_in_ph3'])
    else:
        total_input_power = tip_df.sum().to_frame().T
    total_input_power = total_input_power.round(3)
    write_csv(total_input_power, "sdpf_total_input_power")
    write_csv(losses, "sdpf_power_losses")
    if gen_reg is not None:
        write_csv(gen_reg, "sdpf_distributed_generation")

    # Display summary in console if requested
    if display_summary:
        max_v = ext_v_pu.at[0, 'max'];
        max_bus = ext_v_pu.at[0, 'bus_max']
        min_v = ext_v_pu.at[0, 'min'];
        min_bus = ext_v_pu.at[0, 'bus_min']
        total_P_in = round(total_input_power[['kW_in_ph1', 'kW_in_ph2', 'kW_in_ph3']].iloc[0].sum(), 3)
        total_Q_in = round(total_input_power[['kVAr_in_ph1', 'kVAr_in_ph2', 'kVAr_in_ph3']].iloc[0].sum(), 3)
        total_plosses = round(losses['ploss_totals'].sum() / 1000, 3)
        total_qlosses = round(losses['qloss_totals'].sum() / 1000, 3)
        print(f"maximum voltage (pu): {max_v} at bus {max_bus}")
        print(f"minimum voltage (pu): {min_v} at bus {min_bus}")
        print(f"Total Input Active Power:  {total_P_in} kW")
        print(f"Total Input Reactive Power:  {total_Q_in} kVAr")
        print(f"Total Active Power Losses:  {total_plosses} kW")
        print(f"Total Reactive Power Losses:  {total_qlosses} kVAr\n")
        if gen_reg is not None and not gen_reg.empty:
            print(f"Distributed Generation: {gen_reg}\n")
    print(f"Results in {output_dir or 'current directory'}")
