import numpy as np
import pandas as pd
import shared_state as ST

def forward_backward_sweep(tolerance, max_iterations):
    """
    Runs the forward-backward sweep iterations until convergence or max_iterations is reached.
    Returns a tuple (err_msg, iter_number).
    """
    err_msg = ""
    max_error = 1.0
    iter_number = 0
    # Iterate until the maximum substation voltage error is within tolerance
    while max_error > tolerance:
        iter_number += 1
        forwardsweep()
        backwardsweep()
        # Retrieve substation phase voltages (assumes substation is last in working_buses)
        wb = ST.state.working_buses
        if wb is None or wb.empty:
            err_msg = "No working buses data"
            break
        sub_idx = wb.index[-1]  # index of last row
        v_ph1 = wb.at[sub_idx, 'v_ph1']
        v_ph2 = wb.at[sub_idx, 'v_ph2']
        v_ph3 = wb.at[sub_idx, 'v_ph3']
        sub_phase_voltages = np.array([v_ph1, v_ph2, v_ph3], dtype=complex)
        # Compute substation line-to-line voltages and errors
        sub_line_voltages = ST.state.D.dot(sub_phase_voltages)
        base_line_voltages = ST.state.ELL  # nominal line-to-line voltages
        # Calculate per-phase error (in terms of magnitude difference)
        err1 = abs((abs(base_line_voltages[0]) - abs(sub_line_voltages[0])) / abs(base_line_voltages[0]))
        err2 = abs((abs(base_line_voltages[1]) - abs(sub_line_voltages[1])) / abs(base_line_voltages[1]))
        err3 = abs((abs(base_line_voltages[2]) - abs(sub_line_voltages[2])) / abs(base_line_voltages[2]))
        max_error = max(err1, err2, err3)
        if iter_number == max_iterations:
            err_msg = f"Program halted, maximum number of forward-backward iteration reached ({max_iterations})"
            break
    return err_msg, iter_number

def forwardsweep():
    """
    Executes the forward sweep: starting from the substation, propagate voltages to downstream buses.
    """
    wb = ST.state.working_buses
    gl = ST.state.gen_lines_mat
    # Temporary 3x3 complex matrices
    A = np.zeros((3, 3), dtype=complex)
    B = np.zeros((3, 3), dtype=complex)
    # Set nominal phase voltages at the substation (type 1 bus)
    for n in range(len(wb)):
        if wb.at[n, 'type'] == 1:
            wb.at[n, 'v_ph1'] = ST.state.ELN[0]
            wb.at[n, 'v_ph2'] = ST.state.ELN[1]
            wb.at[n, 'v_ph3'] = ST.state.ELN[2]
            wb.at[n, 'process'] = 1
    # Calculate voltages for each downstream bus
    Vbus1 = ST.state.Vbus1
    Vbus2 = ST.state.Vbus2
    Ibus2 = ST.state.Ibus2
    # Loop through buses in reverse order (substation towards leaves)
    for n in range(len(wb) - 1, -1, -1):
        for m in range(len(gl)):
            if gl.at[m, 'bus1'] == wb.at[n, 'id']:
                # Find the downstream bus with id = gl[m, bus2]
                for n2 in range(len(wb) - 1, -1, -1):
                    if wb.at[n2, 'id'] == gl.at[m, 'bus2']:
                        # Load A and B matrices for this line segment
                        A[0, 0] = gl.at[m, 'A_1_1']; A[0, 1] = gl.at[m, 'A_1_2']; A[0, 2] = gl.at[m, 'A_1_3']
                        A[1, 0] = gl.at[m, 'A_2_1']; A[1, 1] = gl.at[m, 'A_2_2']; A[1, 2] = gl.at[m, 'A_2_3']
                        A[2, 0] = gl.at[m, 'A_3_1']; A[2, 1] = gl.at[m, 'A_3_2']; A[2, 2] = gl.at[m, 'A_3_3']
                        B[0, 0] = gl.at[m, 'B_1_1']; B[0, 1] = gl.at[m, 'B_1_2']; B[0, 2] = gl.at[m, 'B_1_3']
                        B[1, 0] = gl.at[m, 'B_2_1']; B[1, 1] = gl.at[m, 'B_2_2']; B[1, 2] = gl.at[m, 'B_2_3']
                        B[2, 0] = gl.at[m, 'B_3_1']; B[2, 1] = gl.at[m, 'B_3_2']; B[2, 2] = gl.at[m, 'B_3_3']
                        # Form Vbus1 and Ibus2 vectors for calculation
                        Vbus1[0, 0] = wb.at[n, 'v_ph1']
                        Vbus1[1, 0] = wb.at[n, 'v_ph2']
                        Vbus1[2, 0] = wb.at[n, 'v_ph3']
                        Ibus2[0, 0] = wb.at[n2, 'ibus_1']
                        Ibus2[1, 0] = wb.at[n2, 'ibus_2']
                        Ibus2[2, 0] = wb.at[n2, 'ibus_3']
                        # Compute downstream bus voltages: Vbus2 = A * Vbus1 - B * Ibus2
                        Vbus2[:, 0] = A.dot(Vbus1)[:, 0] - B.dot(Ibus2)[:, 0]
                        # Assign calculated voltages to the downstream bus
                        wb.at[n2, 'v_ph1'] = Vbus2[0, 0]
                        wb.at[n2, 'v_ph2'] = Vbus2[1, 0]
                        wb.at[n2, 'v_ph3'] = Vbus2[2, 0]
                        wb.at[n2, 'process'] = 1
    ST.state.working_buses = wb  # update state

def backwardsweep():
    """
    Executes the backward sweep: starting from the ending buses, propagate currents upstream and update voltages.
    """
    wb = ST.state.working_buses
    gl = ST.state.gen_lines_mat
    loads = ST.state.loads
    lines_df = ST.state.lines
    # Temporary 3x3 complex matrices for line calculations
    a_mat = np.zeros((3, 3), dtype=complex)
    b_mat = np.zeros((3, 3), dtype=complex)
    c_mat = np.zeros((3, 3), dtype=complex)
    d_mat = np.zeros((3, 3), dtype=complex)
    # Temporary current vectors (3x1 complex) from shared state
    Iline = ST.state.Iline
    Iphase = ST.state.Iphase
    # Matrix to convert phase currents to line currents for delta loads
    DL = np.array([[1, 0, -1],
                   [-1, 1, 0],
                   [0, -1, 1]], dtype=float)
    # Reset all bus injection currents to 0
    wb[['ibus_1', 'ibus_2', 'ibus_3']] = 0+0j
    # Calculate currents at ending buses (type 5) from their loads
    for n in range(len(wb)):
        if wb.at[n, 'type'] == 5:
            for k in range(len(loads)):
                if loads.at[k, 'bus'] == wb.at[n, 'id']:
                    conn = str(loads.at[k, 'conn']).upper()   # connection type "Y" or "D"
                    ltype = str(loads.at[k, 'type']).upper()  # load type "PQ", "Z", "I", etc.
                    if conn == "Y":
                        # Wye-connected load: add conjugate of (P+jQ)/V or direct I as specified
                        if ltype in ("PQ", "PQV", "PI"):
                            wb.at[n, 'ibus_1'] += np.conj(loads.at[k, 'ph_1'] / wb.at[n, 'v_ph1'])
                            wb.at[n, 'ibus_2'] += np.conj(loads.at[k, 'ph_2'] / wb.at[n, 'v_ph2'])
                            wb.at[n, 'ibus_3'] += np.conj(loads.at[k, 'ph_3'] / wb.at[n, 'v_ph3'])
                        if ltype == "Z":
                            if loads.at[k, 'k_1'] != 0:
                                wb.at[n, 'ibus_1'] += wb.at[n, 'v_ph1'] / loads.at[k, 'k_1']
                            if loads.at[k, 'k_2'] != 0:
                                wb.at[n, 'ibus_2'] += wb.at[n, 'v_ph2'] / loads.at[k, 'k_2']
                            if loads.at[k, 'k_3'] != 0:
                                wb.at[n, 'ibus_3'] += wb.at[n, 'v_ph3'] / loads.at[k, 'k_3']
                        if ltype == "I":
                            wb.at[n, 'ibus_1'] += abs(loads.at[k, 'k_1']) * np.exp(1j * (np.angle(wb.at[n, 'v_ph1']) - np.angle(loads.at[k, 'ph_1'])))
                            wb.at[n, 'ibus_2'] += abs(loads.at[k, 'k_2']) * np.exp(1j * (np.angle(wb.at[n, 'v_ph2']) - np.angle(loads.at[k, 'ph_2'])))
                            wb.at[n, 'ibus_3'] += abs(loads.at[k, 'k_3']) * np.exp(1j * (np.angle(wb.at[n, 'v_ph3']) - np.angle(loads.at[k, 'ph_3'])))
                    elif conn == "D":
                        # Delta-connected load: compute phase currents
                        if ltype in ("PQ", "PQV", "PI"):
                            Iphase[0, 0] = np.conj(loads.at[k, 'ph_1'] / (wb.at[n, 'v_ph1'] - wb.at[n, 'v_ph2']))
                            Iphase[1, 0] = np.conj(loads.at[k, 'ph_2'] / (wb.at[n, 'v_ph2'] - wb.at[n, 'v_ph3']))
                            Iphase[2, 0] = np.conj(loads.at[k, 'ph_3'] / (wb.at[n, 'v_ph3'] - wb.at[n, 'v_ph1']))
                        if ltype == "Z":
                            if loads.at[k, 'k_1'] != 0:
                                Iphase[0, 0] = (wb.at[n, 'v_ph1'] - wb.at[n, 'v_ph2']) / loads.at[k, 'k_1']
                            if loads.at[k, 'k_2'] != 0:
                                Iphase[1, 0] = (wb.at[n, 'v_ph2'] - wb.at[n, 'v_ph3']) / loads.at[k, 'k_2']
                            if loads.at[k, 'k_3'] != 0:
                                Iphase[2, 0] = (wb.at[n, 'v_ph3'] - wb.at[n, 'v_ph1']) / loads.at[k, 'k_3']
                        if ltype == "I":
                            Iphase[0, 0] = abs(loads.at[k, 'k_1']) * np.exp(1j * (np.angle(wb.at[n, 'v_ph1'] - wb.at[n, 'v_ph2']) - np.angle(loads.at[k, 'ph_1'])))
                            Iphase[1, 0] = abs(loads.at[k, 'k_2']) * np.exp(1j * (np.angle(wb.at[n, 'v_ph2'] - wb.at[n, 'v_ph3']) - np.angle(loads.at[k, 'ph_2'])))
                            Iphase[2, 0] = abs(loads.at[k, 'k_3']) * np.exp(1j * (np.angle(wb.at[n, 'v_ph3'] - wb.at[n, 'v_ph1']) - np.angle(loads.at[k, 'ph_3'])))
                        # Convert phase currents to line currents and accumulate
                        Iline[:, 0] = DL.dot(Iphase)[:, 0]
                        wb.at[n, 'ibus_1'] += Iline[0, 0]
                        wb.at[n, 'ibus_2'] += Iline[1, 0]
                        wb.at[n, 'ibus_3'] += Iline[2, 0]
                        # Reset temporary current vectors for next use
                        Iline.fill(0)
                        Iphase.fill(0)
            wb.at[n, 'process'] = 2  # mark end bus as processed
    # Calculate upstream currents and voltages for non-ending buses
    for n in range(len(wb)):
        if wb.at[n, 'type'] != 5:
            # Handle each line where this bus is the sending end (bus1)
            for m in range(len(gl)):
                if gl.at[m, 'bus1'] == wb.at[n, 'id']:
                    # Find the downstream bus (bus2) for this line segment
                    for n2 in range(len(wb)):
                        if wb.at[n2, 'id'] == gl.at[m, 'bus2']:
                            # Load a, b, c, d matrices for the line segment
                            a_mat[0, 0] = gl.at[m, 'a_1_1']; a_mat[0, 1] = gl.at[m, 'a_1_2']; a_mat[0, 2] = gl.at[m, 'a_1_3']
                            a_mat[1, 0] = gl.at[m, 'a_2_1']; a_mat[1, 1] = gl.at[m, 'a_2_2']; a_mat[1, 2] = gl.at[m, 'a_2_3']
                            a_mat[2, 0] = gl.at[m, 'a_3_1']; a_mat[2, 1] = gl.at[m, 'a_3_2']; a_mat[2, 2] = gl.at[m, 'a_3_3']
                            b_mat[0, 0] = gl.at[m, 'b_1_1']; b_mat[0, 1] = gl.at[m, 'b_1_2']; b_mat[0, 2] = gl.at[m, 'b_1_3']
                            b_mat[1, 0] = gl.at[m, 'b_2_1']; b_mat[1, 1] = gl.at[m, 'b_2_2']; b_mat[1, 2] = gl.at[m, 'b_2_3']
                            b_mat[2, 0] = gl.at[m, 'b_3_1']; b_mat[2, 1] = gl.at[m, 'b_3_2']; b_mat[2, 2] = gl.at[m, 'b_3_3']
                            c_mat[0, 0] = gl.at[m, 'c_1_1']; c_mat[0, 1] = gl.at[m, 'c_1_2']; c_mat[0, 2] = gl.at[m, 'c_1_3']
                            c_mat[1, 0] = gl.at[m, 'c_2_1']; c_mat[1, 1] = gl.at[m, 'c_2_2']; c_mat[1, 2] = gl.at[m, 'c_2_3']
                            c_mat[2, 0] = gl.at[m, 'c_3_1']; c_mat[2, 1] = gl.at[m, 'c_3_2']; c_mat[2, 2] = gl.at[m, 'c_3_3']
                            d_mat[0, 0] = gl.at[m, 'd_1_1']; d_mat[0, 1] = gl.at[m, 'd_1_2']; d_mat[0, 2] = gl.at[m, 'd_1_3']
                            d_mat[1, 0] = gl.at[m, 'd_2_1']; d_mat[1, 1] = gl.at[m, 'd_2_2']; d_mat[1, 2] = gl.at[m, 'd_2_3']
                            d_mat[2, 0] = gl.at[m, 'd_3_1']; d_mat[2, 1] = gl.at[m, 'd_3_2']; d_mat[2, 2] = gl.at[m, 'd_3_3']
                            # Form vectors for the downstream bus (bus2) voltage and current
                            Vbus2 = np.array([[wb.at[n2, 'v_ph1']],
                                              [wb.at[n2, 'v_ph2']],
                                              [wb.at[n2, 'v_ph3']]], dtype=complex)
                            Ibus2_vec = np.array([[wb.at[n2, 'ibus_1']],
                                                   [wb.at[n2, 'ibus_2']],
                                                   [wb.at[n2, 'ibus_3']]], dtype=complex)
                            # Compute sending-end bus voltage and current
                            Vbus1_vec = a_mat.dot(Vbus2) + b_mat.dot(Ibus2_vec)
                            Ibus1_vec = c_mat.dot(Vbus2) + d_mat.dot(Ibus2_vec)
                            # Update sending-end bus voltage
                            wb.at[n, 'v_ph1'] = Vbus1_vec[0, 0]
                            wb.at[n, 'v_ph2'] = Vbus1_vec[1, 0]
                            wb.at[n, 'v_ph3'] = Vbus1_vec[2, 0]
                            # Add line current contributions to bus injection currents
                            wb.at[n, 'ibus_1'] += Ibus1_vec[0, 0]
                            wb.at[n, 'ibus_2'] += Ibus1_vec[1, 0]
                            wb.at[n, 'ibus_3'] += Ibus1_vec[2, 0]
                            # Record line sending-end currents in lines DataFrame
                            lines_df.at[m, 'ibus1_1'] = Ibus1_vec[0, 0]
                            lines_df.at[m, 'ibus1_2'] = Ibus1_vec[1, 0]
                            lines_df.at[m, 'ibus1_3'] = Ibus1_vec[2, 0]
                            wb.at[n, 'process'] = 2  # mark bus as processed
            # After processing line flows, add any local load currents at this bus
            for k in range(len(loads)):
                if loads.at[k, 'bus'] == wb.at[n, 'id']:
                    conn = str(loads.at[k, 'conn']).upper()
                    ltype = str(loads.at[k, 'type']).upper()
                    if conn == "Y":
                        if ltype in ("PQ", "PQV", "PI"):
                            wb.at[n, 'ibus_1'] += np.conj(loads.at[k, 'ph_1'] / wb.at[n, 'v_ph1'])
                            wb.at[n, 'ibus_2'] += np.conj(loads.at[k, 'ph_2'] / wb.at[n, 'v_ph2'])
                            wb.at[n, 'ibus_3'] += np.conj(loads.at[k, 'ph_3'] / wb.at[n, 'v_ph3'])
                        if ltype == "Z":
                            if loads.at[k, 'k_1'] != 0:
                                wb.at[n, 'ibus_1'] += wb.at[n, 'v_ph1'] / loads.at[k, 'k_1']
                            if loads.at[k, 'k_2'] != 0:
                                wb.at[n, 'ibus_2'] += wb.at[n, 'v_ph2'] / loads.at[k, 'k_2']
                            if loads.at[k, 'k_3'] != 0:
                                wb.at[n, 'ibus_3'] += wb.at[n, 'v_ph3'] / loads.at[k, 'k_3']
                        if ltype == "I":
                            wb.at[n, 'ibus_1'] += abs(loads.at[k, 'k_1']) * np.exp(1j * (np.angle(wb.at[n, 'v_ph1']) - np.angle(loads.at[k, 'ph_1'])))
                            wb.at[n, 'ibus_2'] += abs(loads.at[k, 'k_2']) * np.exp(1j * (np.angle(wb.at[n, 'v_ph2']) - np.angle(loads.at[k, 'ph_2'])))
                            wb.at[n, 'ibus_3'] += abs(loads.at[k, 'k_3']) * np.exp(1j * (np.angle(wb.at[n, 'v_ph3']) - np.angle(loads.at[k, 'ph_3'])))
                    elif conn == "D":
                        if ltype in ("PQ", "PQV", "PI"):
                            Iphase[0, 0] = np.conj(loads.at[k, 'ph_1'] / (wb.at[n, 'v_ph1'] - wb.at[n, 'v_ph2']))
                            Iphase[1, 0] = np.conj(loads.at[k, 'ph_2'] / (wb.at[n, 'v_ph2'] - wb.at[n, 'v_ph3']))
                            Iphase[2, 0] = np.conj(loads.at[k, 'ph_3'] / (wb.at[n, 'v_ph3'] - wb.at[n, 'v_ph1']))
                        if ltype == "Z":
                            if loads.at[k, 'k_1'] != 0:
                                Iphase[0, 0] = (wb.at[n, 'v_ph1'] - wb.at[n, 'v_ph2']) / loads.at[k, 'k_1']
                            if loads.at[k, 'k_2'] != 0:
                                Iphase[1, 0] = (wb.at[n, 'v_ph2'] - wb.at[n, 'v_ph3']) / loads.at[k, 'k_2']
                            if loads.at[k, 'k_3'] != 0:
                                Iphase[2, 0] = (wb.at[n, 'v_ph3'] - wb.at[n, 'v_ph1']) / loads.at[k, 'k_3']
                        if ltype == "I":
                            Iphase[0, 0] = abs(loads.at[k, 'k_1']) * np.exp(1j * (np.angle(wb.at[n, 'v_ph1'] - wb.at[n, 'v_ph2']) - np.angle(loads.at[k, 'ph_1'])))
                            Iphase[1, 0] = abs(loads.at[k, 'k_2']) * np.exp(1j * (np.angle(wb.at[n, 'v_ph2'] - wb.at[n, 'v_ph3']) - np.angle(loads.at[k, 'ph_2'])))
                            Iphase[2, 0] = abs(loads.at[k, 'k_3']) * np.exp(1j * (np.angle(wb.at[n, 'v_ph3'] - wb.at[n, 'v_ph1']) - np.angle(loads.at[k, 'ph_3'])))
                        # Convert phase currents to line currents and add to bus
                        Iline[:, 0] = DL.dot(Iphase)[:, 0]
                        wb.at[n, 'ibus_1'] += Iline[0, 0]
                        wb.at[n, 'ibus_2'] += Iline[1, 0]
                        wb.at[n, 'ibus_3'] += Iline[2, 0]
                        Iline.fill(0)
                        Iphase.fill(0)
    ST.state.working_buses = wb  # update state
    ST.state.lines = lines_df   # update state
