import numpy as np
import shared_state as ST

def forwardsweep():
    """
    Executes the forward sweep: starting from the substation, propagate voltages to downstream buses.
    """
    wb = ST.state.working_buses.copy()
    gl = ST.state.gen_lines_mat.copy()

    # Temporary 3x3 complex matrices for forward calculation
    A = np.zeros((3, 3), dtype=complex)
    B = np.zeros((3, 3), dtype=complex)

    # Set nominal phase voltages at the substation (type 1 bus)
    for n in range(len(wb)):
        if wb.at[n, 'type'] == 1:
            wb.at[n, 'v_ph1'] = ST.state.ELN[0].copy()
            wb.at[n, 'v_ph2'] = ST.state.ELN[1].copy()
            wb.at[n, 'v_ph3'] = ST.state.ELN[2].copy()
            wb.at[n, 'process'] = 1

    # Calculate voltages for each downstream bus
    Vbus1 = ST.state.Vbus1.copy()  # (3x1) complex
    Vbus2 = ST.state.Vbus2.copy()  # (3x1) complex
    Ibus2 = ST.state.Ibus2.copy()  # (3x1) complex

    # -------------------------------
    # 빠른 해결 1: 변화 없을 때까지 반복
    # -------------------------------
    max_passes = max(1, len(wb))  # 트리 깊이보다 충분히 크게
    for _pass in range(max_passes):
        updates_this_pass = 0

        # Loop through buses in reverse order (substation towards leaves)
        for n in range(len(wb) - 1, -1, -1):
            for m in range(len(gl)):
                if gl.at[m, 'bus1'] != wb.at[n, 'id']:
                    continue

                # Find the downstream bus with id = gl[m, 'bus2']
                for n2 in range(len(wb) - 1, -1, -1):
                    if wb.at[n2, 'id'] != gl.at[m, 'bus2']:
                        continue

                    # Load A and B matrices for this line segment
                    A[0, 0] = gl.at[m, 'A_1_1']
                    A[0, 1] = gl.at[m, 'A_1_2']
                    A[0, 2] = gl.at[m, 'A_1_3']
                    A[1, 0] = gl.at[m, 'A_2_1']
                    A[1, 1] = gl.at[m, 'A_2_2']
                    A[1, 2] = gl.at[m, 'A_2_3']
                    A[2, 0] = gl.at[m, 'A_3_1']
                    A[2, 1] = gl.at[m, 'A_3_2']
                    A[2, 2] = gl.at[m, 'A_3_3']
                    B[0, 0] = gl.at[m, 'B_1_1']
                    B[0, 1] = gl.at[m, 'B_1_2']
                    B[0, 2] = gl.at[m, 'B_1_3']
                    B[1, 0] = gl.at[m, 'B_2_1']
                    B[1, 1] = gl.at[m, 'B_2_2']
                    B[1, 2] = gl.at[m, 'B_2_3']
                    B[2, 0] = gl.at[m, 'B_3_1']
                    B[2, 1] = gl.at[m, 'B_3_2']
                    B[2, 2] = gl.at[m, 'B_3_3']

                    # Form Vbus1 and Ibus2 vectors (parent voltage, child current)
                    Vbus1[0, 0] = wb.at[n, 'v_ph1']
                    Vbus1[1, 0] = wb.at[n, 'v_ph2']
                    Vbus1[2, 0] = wb.at[n, 'v_ph3']
                    Ibus2[0, 0] = wb.at[n2, 'ibus_1']
                    Ibus2[1, 0] = wb.at[n2, 'ibus_2']
                    Ibus2[2, 0] = wb.at[n2, 'ibus_3']

                    # 입력 유효성 검사: 부모 V 또는 자식 I 또는 A/B에 NaN이 있으면 이번 패스에서는 스킵
                    if np.isnan(Vbus1).any():
                        break  # 이 라인의 dn_id 처리는 다음 패스에서 재시도
                    if np.isnan(Ibus2).any():
                        break
                    if np.isnan(A).any() or np.isnan(B).any():
                        break

                    # Compute downstream bus voltages: Vbus2 = A * Vbus1 - B * Ibus2
                    Vbus2[:, 0] = A.dot(Vbus1)[:, 0] - B.dot(Ibus2)[:, 0]

                    # Assign calculated voltages to the downstream bus
                    wb.at[n2, 'v_ph1'] = Vbus2[0, 0]
                    wb.at[n2, 'v_ph2'] = Vbus2[1, 0]
                    wb.at[n2, 'v_ph3'] = Vbus2[2, 0]
                    wb.at[n2, 'process'] = 1
                    updates_this_pass += 1

                    break  # dn_id에 대한 매칭은 완료했으므로 다음 라인(m)으로

        # 더 이상 변화가 없으면 조기 종료
        if updates_this_pass == 0:
            break

    ST.state.working_buses = wb.copy()  # update state after forward sweep

def backwardsweep():
    """
    Executes the backward sweep: starting from the ending buses, propagate currents upstream and update voltages.
    """
    wb = ST.state.working_buses.copy()
    gl = ST.state.gen_lines_mat.copy()
    loads = ST.state.loads.copy()
    lines_df = ST.state.lines.copy()
    # Temporary matrices for backward calculation
    a_mat = np.zeros((3, 3), dtype=complex)
    b_mat = np.zeros((3, 3), dtype=complex)
    c_mat = np.zeros((3, 3), dtype=complex)
    d_mat = np.zeros((3, 3), dtype=complex)
    # Temporary current vectors from shared state
    Iline = ST.state.Iline.copy()
    Iphase = ST.state.Iphase.copy()
    # Matrix to convert phase currents to line currents for delta loads
    DL = np.array([[1, 0, -1],
                   [-1, 1, 0],
                   [0, -1, 1]], dtype=float)
    # Reset all bus injection currents to 0
    wb[['ibus_1', 'ibus_2', 'ibus_3']] = 0 + 0j
    # Calculate currents at ending buses (type 5) from their loads
    for n in range(len(wb)):
        if wb.at[n, 'type'] == 5:
            for k in range(len(loads)):
                if loads.at[k, 'bus'] == wb.at[n, 'id']:
                    conn = str(loads.at[k, 'conn']).upper()
                    ltype = str(loads.at[k, 'type']).upper()
                    if conn == "Y":
                        # Wye-connected load: I = conj(S / V) for PQ/PQV/PI, or direct formulas for Z/I
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
                            wb.at[n, 'ibus_1'] += abs(loads.at[k, 'k_1']) * np.exp(
                                1j * (np.angle(wb.at[n, 'v_ph1']) - np.angle(loads.at[k, 'ph_1'])))
                            wb.at[n, 'ibus_2'] += abs(loads.at[k, 'k_2']) * np.exp(
                                1j * (np.angle(wb.at[n, 'v_ph2']) - np.angle(loads.at[k, 'ph_2'])))
                            wb.at[n, 'ibus_3'] += abs(loads.at[k, 'k_3']) * np.exp(
                                1j * (np.angle(wb.at[n, 'v_ph3']) - np.angle(loads.at[k, 'ph_3'])))
                    elif conn == "D":
                        # Delta-connected load: compute phase currents for each phase of delta
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
                            Iphase[0, 0] = abs(loads.at[k, 'k_1']) * np.exp(
                                1j * (np.angle(wb.at[n, 'v_ph1'] - wb.at[n, 'v_ph2']) - np.angle(loads.at[k, 'ph_1'])))
                            Iphase[1, 0] = abs(loads.at[k, 'k_2']) * np.exp(
                                1j * (np.angle(wb.at[n, 'v_ph2'] - wb.at[n, 'v_ph3']) - np.angle(loads.at[k, 'ph_2'])))
                            Iphase[2, 0] = abs(loads.at[k, 'k_3']) * np.exp(
                                1j * (np.angle(wb.at[n, 'v_ph3'] - wb.at[n, 'v_ph1']) - np.angle(loads.at[k, 'ph_3'])))
                        # Convert phase currents to line currents and accumulate
                        Iline[:, 0] = DL.dot(Iphase)[:, 0]
                        wb.at[n, 'ibus_1'] += Iline[0, 0]
                        wb.at[n, 'ibus_2'] += Iline[1, 0]
                        wb.at[n, 'ibus_3'] += Iline[2, 0]
                        # Reset temporary current vectors
                        Iline.fill(0)
                        Iphase.fill(0)
            wb.at[n, 'process'] = 2  # mark end bus as processed
    # Calculate upstream currents and voltages for non-ending buses
    for n in range(len(wb)):
        if wb.at[n, 'type'] != 5:
            # For each line where this bus is the sending end
            for m in range(len(gl)):
                if gl.at[m, 'bus1'] == wb.at[n, 'id']:
                    # Find the downstream bus (bus2)
                    for n2 in range(len(wb)):
                        if wb.at[n2, 'id'] == gl.at[m, 'bus2']:
                            # Load a, b, c, d matrices for the line
                            a_mat[0, 0] = gl.at[m, 'a_1_1']
                            a_mat[0, 1] = gl.at[m, 'a_1_2']
                            a_mat[0, 2] = gl.at[m, 'a_1_3']
                            a_mat[1, 0] = gl.at[m, 'a_2_1']
                            a_mat[1, 1] = gl.at[m, 'a_2_2']
                            a_mat[1, 2] = gl.at[m, 'a_2_3']
                            a_mat[2, 0] = gl.at[m, 'a_3_1']
                            a_mat[2, 1] = gl.at[m, 'a_3_2']
                            a_mat[2, 2] = gl.at[m, 'a_3_3']
                            b_mat[0, 0] = gl.at[m, 'b_1_1']
                            b_mat[0, 1] = gl.at[m, 'b_1_2']
                            b_mat[0, 2] = gl.at[m, 'b_1_3']
                            b_mat[1, 0] = gl.at[m, 'b_2_1']
                            b_mat[1, 1] = gl.at[m, 'b_2_2']
                            b_mat[1, 2] = gl.at[m, 'b_2_3']
                            b_mat[2, 0] = gl.at[m, 'b_3_1']
                            b_mat[2, 1] = gl.at[m, 'b_3_2']
                            b_mat[2, 2] = gl.at[m, 'b_3_3']
                            c_mat[0, 0] = gl.at[m, 'c_1_1']
                            c_mat[0, 1] = gl.at[m, 'c_1_2']
                            c_mat[0, 2] = gl.at[m, 'c_1_3']
                            c_mat[1, 0] = gl.at[m, 'c_2_1']
                            c_mat[1, 1] = gl.at[m, 'c_2_2']
                            c_mat[1, 2] = gl.at[m, 'c_2_3']
                            c_mat[2, 0] = gl.at[m, 'c_3_1']
                            c_mat[2, 1] = gl.at[m, 'c_3_2']
                            c_mat[2, 2] = gl.at[m, 'c_3_3']
                            d_mat[0, 0] = gl.at[m, 'd_1_1']
                            d_mat[0, 1] = gl.at[m, 'd_1_2']
                            d_mat[0, 2] = gl.at[m, 'd_1_3']
                            d_mat[1, 0] = gl.at[m, 'd_2_1']
                            d_mat[1, 1] = gl.at[m, 'd_2_2']
                            d_mat[1, 2] = gl.at[m, 'd_2_3']
                            d_mat[2, 0] = gl.at[m, 'd_3_1']
                            d_mat[2, 1] = gl.at[m, 'd_3_2']
                            d_mat[2, 2] = gl.at[m, 'd_3_3']
                            # Downstream bus voltage and current vectors
                            Vbus2_vec = np.array([[wb.at[n2, 'v_ph1']],
                                                  [wb.at[n2, 'v_ph2']],
                                                  [wb.at[n2, 'v_ph3']]], dtype=complex)
                            Ibus2_vec = np.array([[wb.at[n2, 'ibus_1']],
                                                  [wb.at[n2, 'ibus_2']],
                                                  [wb.at[n2, 'ibus_3']]], dtype=complex)
                            # Compute sending-end bus voltage and current
                            Vbus1_vec = a_mat.dot(Vbus2_vec) + b_mat.dot(Ibus2_vec)
                            Ibus1_vec = c_mat.dot(Vbus2_vec) + d_mat.dot(Ibus2_vec)
                            # Update sending-end bus voltage and injection current
                            wb.at[n, 'v_ph1'] = Vbus1_vec[0, 0]
                            wb.at[n, 'v_ph2'] = Vbus1_vec[1, 0]
                            wb.at[n, 'v_ph3'] = Vbus1_vec[2, 0]
                            wb.at[n, 'ibus_1'] += Ibus1_vec[0, 0]
                            wb.at[n, 'ibus_2'] += Ibus1_vec[1, 0]
                            wb.at[n, 'ibus_3'] += Ibus1_vec[2, 0]
                            # Record sending-end line currents
                            lines_df.at[m, 'ibus1_1'] = Ibus1_vec[0, 0]
                            lines_df.at[m, 'ibus1_2'] = Ibus1_vec[1, 0]
                            lines_df.at[m, 'ibus1_3'] = Ibus1_vec[2, 0]
                            wb.at[n, 'process'] = 2
            # After processing line flows, add local load currents at this bus (if any)
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
                            wb.at[n, 'ibus_1'] += abs(loads.at[k, 'k_1']) * np.exp(
                                1j * (np.angle(wb.at[n, 'v_ph1']) - np.angle(loads.at[k, 'ph_1'])))
                            wb.at[n, 'ibus_2'] += abs(loads.at[k, 'k_2']) * np.exp(
                                1j * (np.angle(wb.at[n, 'v_ph2']) - np.angle(loads.at[k, 'ph_2'])))
                            wb.at[n, 'ibus_3'] += abs(loads.at[k, 'k_3']) * np.exp(
                                1j * (np.angle(wb.at[n, 'v_ph3']) - np.angle(loads.at[k, 'ph_3'])))
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
                            Iphase[0, 0] = abs(loads.at[k, 'k_1']) * np.exp(
                                1j * (np.angle(wb.at[n, 'v_ph1'] - wb.at[n, 'v_ph2']) - np.angle(loads.at[k, 'ph_1'])))
                            Iphase[1, 0] = abs(loads.at[k, 'k_2']) * np.exp(
                                1j * (np.angle(wb.at[n, 'v_ph2'] - wb.at[n, 'v_ph3']) - np.angle(loads.at[k, 'ph_2'])))
                            Iphase[2, 0] = abs(loads.at[k, 'k_3']) * np.exp(
                                1j * (np.angle(wb.at[n, 'v_ph3'] - wb.at[n, 'v_ph1']) - np.angle(loads.at[k, 'ph_3'])))
                        # Convert phase currents to line currents and add to bus
                        Iline[:, 0] = DL.dot(Iphase)[:, 0]
                        wb.at[n, 'ibus_1'] += Iline[0, 0]
                        wb.at[n, 'ibus_2'] += Iline[1, 0]
                        wb.at[n, 'ibus_3'] += Iline[2, 0]
                        Iline.fill(0)
                        Iphase.fill(0)
    ST.state.working_buses = wb.copy()  # update state after backward sweep
    ST.state.lines = lines_df.copy()  # update state after backward sweep

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
        wb = ST.state.working_buses.copy()
        if wb is None or wb.empty:
            err_msg = "No working buses data"
            break
        sub_idx = wb.index[-1]  # index of last row
        v_ph1 = wb.at[sub_idx, 'v_ph1']
        v_ph2 = wb.at[sub_idx, 'v_ph2']
        v_ph3 = wb.at[sub_idx, 'v_ph3']
        sub_phase_voltages = np.array([v_ph1, v_ph2, v_ph3], dtype=complex)
        # Compute substation line-to-line voltages and errors
        sub_line_voltages = ST.state.D.dot(sub_phase_voltages).copy()
        base_line_voltages = ST.state.ELL.copy()  # nominal line-to-line voltages
        # Calculate per-phase error (in terms of magnitude difference)
        err1 = abs((abs(base_line_voltages[0]) - abs(sub_line_voltages[0])) / abs(base_line_voltages[0]))
        err2 = abs((abs(base_line_voltages[1]) - abs(sub_line_voltages[1])) / abs(base_line_voltages[1]))
        err3 = abs((abs(base_line_voltages[2]) - abs(sub_line_voltages[2])) / abs(base_line_voltages[2]))
        max_error = max(err1, err2, err3)
        if iter_number == max_iterations:
            err_msg = f"Program halted, maximum number of forward-backward iteration reached ({max_iterations})"
            break
    return err_msg, iter_number
