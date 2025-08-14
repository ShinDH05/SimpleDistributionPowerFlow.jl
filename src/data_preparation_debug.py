import math
import numpy as np
import pandas as pd
import shared_state as ST
from topology_discovery import gridtopology

# Absolute input directory for case files
input_dir = r"C:\dev\ACPF\BFS\examples\ieee-123"
caller = "powerflow"

# Build network topology and populate shared_state using the input data
gridtopology(caller, input_dir)

# 1. Initialize generation-related global flags and DataFrames
ST.state.has_pq_distributed_gen = False
ST.state.pq_distributed_gen = pd.DataFrame()
ST.state.has_pqv_distributed_gen = False
ST.state.pqv_distributed_gen = pd.DataFrame()
ST.state.has_pi_distributed_gen = False
ST.state.pi_distributed_gen = pd.DataFrame()
ST.state.generation_register = pd.DataFrame()
err_msg = ""

# 2. Classify lines and buses (populate working_buses types/numbers and lines types)
# Each segment is classified by type (1=line, 2=transformer, 3=switch, 4=regulator).
# Each bus is classified by type (1=substation, 2=bifurcation, 3=intermediate, 4=next-to-end, 5=end).
# Mark buses downstream of transformers using the 'trf' field.
wb = ST.state.working_buses.copy()         # DataFrame of working buses (must have 'id' column)
ws = ST.state.working_segments.copy()      # DataFrame of working segments (columns: bus1, bus2, length, unit, config, etc.)
# Initialize lines DataFrame by copying working_segments and adding 'type' column
lines_df = ws.copy()
lines_df['type'] = 0  # default type 0
# Determine type of each line segment based on its config
for idx in lines_df.index:
    cfg = lines_df.at[idx, 'config']
    if not ST.state.line_configurations.empty and (cfg in ST.state.line_configurations['config'].values):
        lines_df.at[idx, 'type'] = 1  # line segment
    if ST.state.has_transformer and (cfg in ST.state.transformers['config'].values):
        lines_df.at[idx, 'type'] = 2  # inline transformer
    if ST.state.has_switch and (cfg in ST.state.switches['config'].values):
        lines_df.at[idx, 'type'] = 3  # switch
    if ST.state.has_regulator and (cfg in ST.state.regulators['config'].values):
        lines_df.at[idx, 'type'] = 4  # regulator

# Add 'type' and 'number' columns to working_buses (initialize to 0)
wb['type'] = 0
wb['number'] = 0

# Compute adjacency (downstream) matrix for working_buses to identify child counts
bus_ids = wb['id'].values
index_map = {bus_id: i for i, bus_id in enumerate(bus_ids)}
n_buses = len(wb)
adj_matrix = np.zeros((n_buses, n_buses), dtype=int)
for _, seg in ws.iterrows():
    b1 = seg['bus1']; b2 = seg['bus2']
    if b1 in index_map and b2 in index_map:
        i = index_map[b1]; j = index_map[b2]
        adj_matrix[i, j] = 1
downward_counts = adj_matrix.sum(axis=1)

# Assign end (5) and bifurcation (2) bus types based on child counts
for i in range(n_buses):
    if downward_counts[i] == 0:
        wb.at[i, 'type'] = 5  # end bus (no children)
    elif downward_counts[i] > 1:
        wb.at[i, 'type'] = 2  # bifurcation bus (more than one child)

# Map each bus to its list of children for quick lookup
children_map = {}
for _, seg in ws.iterrows():
    children_map.setdefault(seg['bus1'], []).append(seg['bus2'])

# Assign intermediate (3) vs next-to-end (4) for buses with exactly one child
for i in range(n_buses):
    if downward_counts[i] == 1:
        cur_id = wb.at[i, 'id']
        child_id = children_map.get(cur_id, [None])[0]
        if child_id is None:
            continue
        child_idx = index_map.get(child_id, None)
        child_type = wb.at[child_idx, 'type'] if child_idx is not None else 0
        if child_type == 5:
            wb.at[i, 'type'] = 4  # next-to-end bus (single child is an end bus)
        else:
            wb.at[i, 'type'] = 3  # intermediate bus (single child is not an end)

# Mark the substation bus type (1) for the substation (root bus)
if ST.state.substation is not None:
    sub_bus = int(ST.state.substation.iloc[0]['bus'])
    if sub_bus in index_map:
        wb.at[index_map[sub_bus], 'type'] = 1

# Sort working_buses by type so that substation (1) comes first, etc., then reset index
wb.sort_values(by='type', ascending=True, inplace=True)
wb.reset_index(drop=True, inplace=True)

# Assign sequential numbers to substation (1), next-to-end (4), and end (5) buses in sorted order
num_counter = 1
for i in range(len(wb)):
    if wb.at[i, 'type'] in (1, 4, 5):
        wb.at[i, 'number'] = num_counter
        num_counter += 1

# Count how many bifurcation, next-to-end, and end buses
bifurcation_count = (wb['type'] == 2).sum()
next_to_end_count = (wb['type'] == 4).sum()
end_count = (wb['type'] == 5).sum()

# Starting number for intermediate and bifurcation buses (they will get the highest numbers)
k = len(wb) - end_count - next_to_end_count
if bifurcation_count > 0:
    # Iteratively assign intermediate bus numbers, then bifurcation bus numbers, ensuring children are numbered first
    for _ in range(bifurcation_count):
        # Pass 1: assign intermediate (type 3) buses whose children have been numbered
        for _ in range(next_to_end_count):
            for i in range(len(wb)):
                if wb.at[i, 'type'] == 3 and wb.at[i, 'number'] == 0:
                    cur_id = wb.at[i, 'id']
                    for j in range(len(wb)):
                        if wb.at[j, 'id'] in children_map.get(cur_id, []) and wb.at[j, 'number'] != 0:
                            wb.at[i, 'number'] = k
                            k -= 1
                            break
        # Pass 2: assign bifurcation (type 2) buses whose all children are numbered
        for i in range(len(wb)):
            if wb.at[i, 'type'] == 2 and wb.at[i, 'number'] == 0:
                cur_id = wb.at[i, 'id']
                waiting = False
                for child in children_map.get(cur_id, []):
                    child_idx = index_map.get(child, None)
                    if child_idx is not None and wb.at[child_idx, 'number'] == 0:
                        waiting = True
                        break
                if not waiting:
                    wb.at[i, 'number'] = k
                    k -= 1
else:
    # If there are no bifurcations, directly assign remaining intermediate buses (type 3) based on their child’s number
    for i in range(len(wb)):
        if wb.at[i, 'type'] == 3 and wb.at[i, 'number'] == 0:
            cur_id = wb.at[i, 'id']
            for child in children_map.get(cur_id, []):
                child_idx = index_map.get(child, None)
                if child_idx is not None and wb.at[child_idx, 'number'] != 0:
                    wb.at[i, 'number'] = wb.at[child_idx, 'number'] - 1
                    break

# Assign any remaining intermediate buses by propagating from their parent's number (in case any left unnumbered)
unnumbered = wb[(wb['type'] == 3) & (wb['number'] == 0)]
while not unnumbered.empty:
    for i in unnumbered.index:
        cur_id = wb.at[i, 'id']
        parent_seg = ws[ws['bus2'] == cur_id]
        if not parent_seg.empty:
            parent_id = int(parent_seg.iloc[0]['bus1'])
            parent_idx = index_map.get(parent_id, None)
            if parent_idx is not None:
                parent_num = wb.at[parent_idx, 'number']
                if parent_num > 0:
                    wb.at[i, 'number'] = parent_num + 1
    unnumbered = wb[(wb['type'] == 3) & (wb['number'] == 0)]

# Sort working_buses by the assigned number (ascending) for final consistent ordering
wb.sort_values(by='number', ascending=True, inplace=True)
wb.reset_index(drop=True, inplace=True)

def rebuild_index_map(wb: pd.DataFrame) -> dict:
    ids = pd.to_numeric(wb['id'], errors='coerce')
    return {int(ids.iloc[i]): int(i) for i in range(len(wb)) if not pd.isna(ids.iloc[i])}

index_map = rebuild_index_map(wb)

# Add a 'trf' column to working_buses to mark buses downstream of transformers
wb['trf'] = None
if ST.state.has_transformer:
    # Mark the immediate downstream bus of each transformer segment
    for _, tr in ST.state.transformers.iterrows():
        cfg = tr['config']
        trans_seg = lines_df[lines_df['config'] == cfg]
        for _, seg in trans_seg.iterrows():
            bus2_id = seg['bus2']
            if bus2_id in index_map:
                wb.at[index_map[bus2_id], 'trf'] = cfg
    # Propagate the transformer flag to all children of those buses (mark entire downstream of a transformer)
    for i in range(len(wb)):
        if wb.at[i, 'trf'] is not None and wb.at[i, 'type'] != 5:
            cur_id = wb.at[i, 'id']
            for child in children_map.get(cur_id, []):
                child_idx = index_map.get(child, None)
                if child_idx is not None:
                    wb.at[child_idx, 'trf'] = wb.at[i, 'trf']

# Save classification results back to shared state for use in subsequent steps
ST.state.working_buses = wb.copy()
ST.state.lines = lines_df.copy()

# 3. Compute base nominal voltages and transformation matrices from substation data
substation_df = ST.state.substation
if substation_df is None or substation_df.empty:
    err_msg = "Substation data not available"
    raise SystemExit(err_msg)
base_kv = float(substation_df.iloc[0]['kv'])      # substation base voltage (line-to-line, in kV)
ell = base_kv * 1000.0                            # nominal line-to-line voltage in volts
eln = ell / math.sqrt(3)                          # nominal line-to-neutral voltage in volts
# Complex base phase-to-neutral voltages (120° apart)
ELN = np.array([
    eln,
    eln * np.exp(-1j * np.deg2rad(120)),
    eln * np.exp(1j * np.deg2rad(120))
], dtype=complex)
# Matrix to convert phase voltages to line-to-line voltages
D_matrix = np.array([[1, -1, 0],
                     [0, 1, -1],
                     [-1, 0, 1]], dtype=float)
ELL = D_matrix.dot(ELN)  # base line-to-line voltages (complex)
# Store base voltages and transformation matrices in shared state
alpha = np.exp(1j * np.deg2rad(120))  # e^(j120°), for sequence matrix
As_matrix = np.array([[1,        1,       1],
                      [1, alpha**2, alpha    ],
                      [1, alpha,    alpha**2]], dtype=complex)
ST.state.ELN = ELN
ST.state.D = D_matrix
ST.state.ELL = ELL
ST.state.As = As_matrix

# 4. Add base voltage (v_base) to each working bus (line-to-neutral base voltage in volts)
wb = ST.state.working_buses.copy()
wb['v_base'] = 0.0
for i in range(len(wb)):
    trf_cfg = wb.at[i, 'trf']
    if trf_cfg is not None:
        # If this bus is downstream of a transformer, use that transformer's low-side base voltage
        for _, tr in ST.state.transformers.iterrows():
            if tr['config'] == trf_cfg:
                wb.at[i, 'v_base'] = tr['kv_low'] * 1000.0 / math.sqrt(3)
                break
    else:
        # Otherwise, use system base line-to-neutral voltage
        wb.at[i, 'v_base'] = eln
ST.state.working_buses = wb.copy()  # update shared state with v_base values

# 5. Construct working line_configurations DataFrame with complex impedances per unit length
lc = ST.state.line_configurations.copy()
if not lc.empty:
    # Start with config and unit columns
    line_configs = lc[['config', 'unit']].copy()
    # Complex impedances (Ohm per unit length) for each configuration
    line_configs['zaa'] = lc['raa'] + 1j * lc['xaa']
    line_configs['zab'] = lc['rab'] + 1j * lc['xab']
    line_configs['zac'] = lc['rac'] + 1j * lc['xac']
    line_configs['zbb'] = lc['rbb'] + 1j * lc['xbb']
    line_configs['zbc'] = lc['rbc'] + 1j * lc['xbc']
    line_configs['zcc'] = lc['rcc'] + 1j * lc['xcc']
    # Shunt susceptance (imaginary admittance) per unit length: convert given B values to jB
    for col in ['baa', 'bab', 'bac', 'bbb', 'bbc', 'bcc']:
        line_configs[col] = 1j * lc[col]
else:
    # If no line configuration data, create an empty DataFrame with expected columns
    line_configs = pd.DataFrame(columns=['config','unit','zaa','zab','zac',
                                        'zbb','zbc','zcc','baa','bab','bac','bbb','bbc','bcc'])
ST.state.line_configs = line_configs.copy()

# 6. Extend lines DataFrame with 'phases' and impedance columns (initialize to 0 or None)
lines = ST.state.lines.copy()  # DataFrame of line segments from working_segments (set in classification step)
lines['phases'] = None
for col in ['Zaa','Zab','Zac','Zbb','Zbc','Zcc','Baa','Bab','Bac','Bbb','Bbc','Bcc']:
    lines[col] = 0+0j  # initialize all impedance (Z) and admittance (B) entries to complex 0

# 7. Compute transformer complex impedance (Zt) if transformers exist
if ST.state.has_transformer:
    ST.state.transformers['Zt'] = ((ST.state.transformers['kv_low'] ** 2 / ST.state.transformers['kva']) *
                                   (ST.state.transformers['rpu'] + 1j * ST.state.transformers['xpu']) * 1000.0)

# 8. Impedance matrices construction for each line segment
for idx in lines.index:
    seg_type = lines.at[idx, 'type']
    cfg = lines.at[idx, 'config']
    if seg_type == 1:
        # Line segment: calculate series impedance and shunt admittance based on length
        factor = 1.0
        cfg_row = line_configs[line_configs['config'] == cfg]
        if not cfg_row.empty:
            cfg_unit = cfg_row.iloc[0]['unit']
            line_unit = lines.at[idx, 'unit']
            # Determine unit conversion factor (to match config data units)
            if line_unit == "ft" and cfg_unit == "mi":
                factor = 1/5280.0
            elif line_unit == "m" and cfg_unit == "km":
                factor = 1/1000.0
            elif line_unit == "m" and cfg_unit == "mi":
                factor = 1/1609.344
            elif line_unit == "ft" and cfg_unit == "km":
                factor = 1/3280.8399
            length = lines.at[idx, 'length']
            # Compute per-segment series impedance (Ohm) = Z_per_unit_length * length * conversion_factor
            lines.at[idx, 'Zaa'] = cfg_row.iloc[0]['zaa'] * length * factor
            lines.at[idx, 'Zab'] = cfg_row.iloc[0]['zab'] * length * factor
            lines.at[idx, 'Zac'] = cfg_row.iloc[0]['zac'] * length * factor
            lines.at[idx, 'Zbb'] = cfg_row.iloc[0]['zbb'] * length * factor
            lines.at[idx, 'Zbc'] = cfg_row.iloc[0]['zbc'] * length * factor
            lines.at[idx, 'Zcc'] = cfg_row.iloc[0]['zcc'] * length * factor
            # Compute per-segment shunt susceptance (S) = B_microSiemens_per_unit_length * length * conversion * 1e-6
            lines.at[idx, 'Baa'] = cfg_row.iloc[0]['baa'] * length * factor * 1e-6
            lines.at[idx, 'Bab'] = cfg_row.iloc[0]['bab'] * length * factor * 1e-6
            lines.at[idx, 'Bac'] = cfg_row.iloc[0]['bac'] * length * factor * 1e-6
            lines.at[idx, 'Bbb'] = cfg_row.iloc[0]['bbb'] * length * factor * 1e-6
            lines.at[idx, 'Bbc'] = cfg_row.iloc[0]['bbc'] * length * factor * 1e-6
            lines.at[idx, 'Bcc'] = cfg_row.iloc[0]['bcc'] * length * factor * 1e-6
        # Determine phase configuration string based on which diagonal impedances are non-zero
        Zaa = lines.at[idx, 'Zaa']; Zbb = lines.at[idx, 'Zbb']; Zcc = lines.at[idx, 'Zcc']
        phase_str = ""
        if Zaa == 0 and Zbb == 0 and Zcc != 0: phase_str = "c"
        if Zaa == 0 and Zbb != 0 and Zcc == 0: phase_str = "b"
        if Zaa != 0 and Zbb == 0 and Zcc == 0: phase_str = "a"
        if Zaa != 0 and Zbb != 0 and Zcc == 0: phase_str = "ab"
        if Zaa != 0 and Zbb == 0 and Zcc != 0: phase_str = "ac"
        if Zaa == 0 and Zbb != 0 and Zcc != 0: phase_str = "bc"
        if Zaa != 0 and Zbb != 0 and Zcc != 0: phase_str = "abc"
        lines.at[idx, 'phases'] = phase_str

    elif seg_type == 2:
        # Transformer segment
        tr = ST.state.transformers[ST.state.transformers['config'] == cfg]
        if tr.empty:
            continue  # no matching transformer data (shouldn't happen if input is consistent)
        Zt_val = tr.iloc[0]['Zt']
        # Use transformer impedance for self-impedances, zero for mutuals and admittances
        lines.at[idx, 'Zaa'] = Zt_val
        lines.at[idx, 'Zbb'] = Zt_val
        lines.at[idx, 'Zcc'] = Zt_val
        lines.at[idx, 'Zab'] = 0+0j; lines.at[idx, 'Zac'] = 0+0j; lines.at[idx, 'Zbc'] = 0+0j
        lines.at[idx, 'Baa'] = 0+0j; lines.at[idx, 'Bab'] = 0+0j; lines.at[idx, 'Bac'] = 0+0j
        lines.at[idx, 'Bbb'] = 0+0j; lines.at[idx, 'Bbc'] = 0+0j; lines.at[idx, 'Bcc'] = 0+0j
        lines.at[idx, 'phases'] = tr.iloc[0]['phases'] if 'phases' in tr.columns else "abc"

    elif seg_type == 3:
        # Switch segment
        sw = ST.state.switches[ST.state.switches['config'] == cfg]
        if sw.empty:
            continue  # no matching switch config (shouldn't happen if input is consistent)
        if sw.iloc[0]['state'].upper() == "CLOSED":
            res = sw.iloc[0]['resistance']
            # Closed switch: small resistance on each phase
            lines.at[idx, 'Zaa'] = res
            lines.at[idx, 'Zbb'] = res
            lines.at[idx, 'Zcc'] = res
        else:
            # Open switch: infinite impedance on each phase
            inf_val = float('inf')
            lines.at[idx, 'Zaa'] = inf_val
            lines.at[idx, 'Zbb'] = inf_val
            lines.at[idx, 'Zcc'] = inf_val
        # No coupling impedances or shunt admittances for switches
        lines.at[idx, 'Zab'] = 0+0j; lines.at[idx, 'Zac'] = 0+0j; lines.at[idx, 'Zbc'] = 0+0j
        lines.at[idx, 'Baa'] = 0+0j; lines.at[idx, 'Bab'] = 0+0j; lines.at[idx, 'Bac'] = 0+0j
        lines.at[idx, 'Bbb'] = 0+0j; lines.at[idx, 'Bbc'] = 0+0j; lines.at[idx, 'Bcc'] = 0+0j
        lines.at[idx, 'phases'] = sw.iloc[0]['phases'] if 'phases' in sw.columns else "abc"

    elif seg_type == 4:
        # Regulator segment (manual mode assumed, ideal regulator with no impedance)
        reg = ST.state.regulators[ST.state.regulators['config'] == cfg]
        if reg.empty:
            # If no regulator data found, leave default identity (no change)
            pass
        # All Z and B entries remain 0; just assign phase config if available
        lines.at[idx, 'phases'] = reg.iloc[0]['phases'] if not reg.empty else "abc"

# Update the shared state lines DataFrame with filled impedance values
ST.state.lines = lines.copy()

# 9. Construct generalized line matrices (a, b, c, d, A, B) for each line
gen_cols = ['bus1','bus2'] + [f"{m}_{p}_{q}" for m in ['a','b','c','d','A','B'] for p in range(1,4) for q in range(1,4)]
gen_lines_list = []
I3 = np.eye(3, dtype=complex)
for _, line in lines.iterrows():
    m_type = line['type']
    # Build per-line impedance (Z) and admittance (Y) matrices from the line data
    z_line = np.array([[line['Zaa'], line['Zab'], line['Zac']],
                       [line['Zab'], line['Zbb'], line['Zbc']],
                       [line['Zac'], line['Zbc'], line['Zcc']]], dtype=complex)
    y_line = np.array([[line['Baa'], line['Bab'], line['Bac']],
                       [line['Bab'], line['Bbb'], line['Bbc']],
                       [line['Bac'], line['Bbc'], line['Bcc']]], dtype=complex)
    a_mat = b_mat = c_mat = d_mat = A_mat = B_mat = None
    if m_type == 1 or m_type == 3:
        # Line or Switch (treat switch like a line with zero/negligible impedance)
        a_mat = I3 + 0.5 * z_line.dot(y_line)
        b_mat = z_line
        c_mat = y_line + 0.25 * y_line.dot(z_line).dot(y_line)
        d_mat = I3 + 0.5 * y_line.dot(z_line)
        A_mat = np.linalg.inv(a_mat)
        B_mat = A_mat.dot(b_mat)
    elif m_type == 2:
        # Transformer: use Kersting's formulas based on winding connections
        tr = ST.state.transformers[ST.state.transformers['config'] == line['config']]
        if tr.empty:
            continue  # should not happen if transformer data exists
        ch = tr.iloc[0]['conn_high']; cl = tr.iloc[0]['conn_low']; Zt = tr.iloc[0]['Zt']
        if (ch == "GRY" and cl == "GRY") or (ch == "D" and cl == "D"):
            # Grounded Wye - Grounded Wye OR Delta - Delta connection
            nt = tr.iloc[0]['kv_high'] / tr.iloc[0]['kv_low']
            a_mat = nt * I3
            b_mat = a_mat * Zt
            c_mat = np.zeros((3,3), complex)
            d_mat = (1/nt) * I3
            A_mat = d_mat.copy()   # a_mat = nt*I, so A_mat = 1/nt * I = d_mat
            B_mat = np.diag([Zt, Zt, Zt])
        elif ch == "D" and cl == "GRY":
            # Delta - Grounded Wye transformer (high side Δ, low side Y-gnd)
            nt = math.sqrt(3) * tr.iloc[0]['kv_high'] / tr.iloc[0]['kv_low']
            a_mat = (-nt/3) * np.array([[0, 2, 1],
                                        [1, 0, 2],
                                        [2, 1, 0]], dtype=complex)
            b_mat = a_mat * Zt
            c_mat = np.zeros((3,3), complex)
            d_mat = (1/nt) * np.array([[1, -1, 0],
                                       [0, 1, -1],
                                       [-1, 0, 1]], dtype=complex)
            A_mat = (1/nt) * np.array([[1, 0, -1],
                                       [-1, 1, 0],
                                       [0, -1, 1]], dtype=complex)
            B_mat = np.diag([Zt, Zt, Zt])
        elif ch == "Y" and cl == "D":
            # Ungrounded Wye - Delta transformer (high side Y-ungrounded, low side Δ)
            nt = (tr.iloc[0]['kv_high'] / math.sqrt(3)) / tr.iloc[0]['kv_low']
            a_mat = nt * np.array([[1, -1, 0],
                                    [0, 1, -1],
                                    [-1, 0, 1]], dtype=complex)
            b_mat = nt * np.array([[1, -1, 0],
                                   [1,  2, 0],
                                   [-2, -1, 0]], dtype=complex) * Zt
            c_mat = np.zeros((3,3), complex)
            d_mat = (1/(3*nt)) * np.array([[1, -1, 0],
                                           [1,  2, 0],
                                           [-2, -1, 0]], dtype=complex)
            A_mat = (1/(3*nt)) * np.array([[2, 1, 0],
                                           [0, 2, 1],
                                           [1, 0, 2]], dtype=complex)
            B_mat = np.array([[ Zt,   0,   0],
                              [  0,  Zt,   0],
                              [ -Zt, -Zt,   0]], dtype=complex)
        else:
            # Unsupported transformer configuration encountered
            err_msg = "revise transformers.csv file, unsupported transformer configuration."
            raise SystemExit(err_msg)
    elif m_type == 4:
        # Regulator (Manual mode) – assumed ideal (a*d = I, b=c=0)
        reg = ST.state.regulators[ST.state.regulators['config'] == line['config']]
        b_mat = np.zeros((3, 3), dtype=complex)
        c_mat = np.zeros((3, 3), dtype=complex)

        use_manual = False
        if not reg.empty:
            if 'mode' in reg.columns:
                use_manual = (str(reg.iloc[0]['mode']).upper() == 'MANUAL')
            else:
                use_manual = True

        if use_manual and not reg.empty:
            tap1 = reg.iloc[0]['tap_1']
            tap2 = reg.iloc[0]['tap_2']
            tap3 = reg.iloc[0]['tap_3']
            a_mat = np.diag([1 / (1 + 0.00625 * tap1),
                             1 / (1 + 0.00625 * tap2),
                             1 / (1 + 0.00625 * tap3)])
            d_mat = np.diag([1 + 0.00625 * tap1,
                             1 + 0.00625 * tap2,
                             1 + 0.00625 * tap3])
        else:
            a_mat = I3
            d_mat = I3
        A_mat = d_mat.copy()  # For an ideal regulator, A = d since a * d = I
        B_mat = np.zeros((3,3), complex)
    # Flatten the matrices into one row of gen_lines_mat
    gen_entry = {'bus1': line['bus1'], 'bus2': line['bus2']}
    for mat, prefix in [(a_mat, 'a'), (b_mat, 'b'), (c_mat, 'c'), (d_mat, 'd'), (A_mat, 'A'), (B_mat, 'B')]:
        for p in range(3):
            for q in range(3):
                gen_entry[f"{prefix}_{p+1}_{q+1}"] = mat[p, q] if mat is not None else None
    gen_lines_list.append(gen_entry)
# Create DataFrame for generalized line matrices and store in shared state
gen_lines_mat_df = pd.DataFrame(gen_lines_list, columns=gen_cols)
ST.state.gen_lines_mat = gen_lines_mat_df.copy()

# 10. Build consolidated loads list (spot loads, distributed loads, capacitors, etc.)
loads_list = []
# Spot loads
if ST.state.spot_loads is not None:
    for _, sl in ST.state.spot_loads.iterrows():
        bus_id = sl['bus']
        if bus_id in wb['id'].values:
            s_ph1 = (sl['kw_ph1'] + 1j*sl['kvar_ph1']) * 1000  # convert kW, kVAr to VA (complex)
            s_ph2 = (sl['kw_ph2'] + 1j*sl['kvar_ph2']) * 1000
            s_ph3 = (sl['kw_ph3'] + 1j*sl['kvar_ph3']) * 1000
            loads_list.append({
                'bus': bus_id,
                'conn': sl['conn'],
                'type': sl['type'],
                'ph_1': s_ph1, 'ph_2': s_ph2, 'ph_3': s_ph3
            })
# Distributed loads (treated as a spot load at the start bus of each segment that has a distributed load)
if ST.state.has_distributed_load and ST.state.distributed_loads is not None:
    for _, dl in ST.state.distributed_loads.iterrows():
        seg = ST.state.working_segments[ST.state.working_segments['bus2'] == dl['bus2']]
        if seg.empty:
            continue  # segment not found (should not happen if input is consistent)
        start_bus = int(seg.iloc[0]['bus1'])
        s_ph1 = (dl['kw_ph1'] + 1j*dl['kvar_ph1']) * 1000
        s_ph2 = (dl['kw_ph2'] + 1j*dl['kvar_ph2']) * 1000
        s_ph3 = (dl['kw_ph3'] + 1j*dl['kvar_ph3']) * 1000
        loads_list.append({
            'bus': start_bus,
            'conn': dl['conn'],
            'type': dl['type'],
            'ph_1': s_ph1, 'ph_2': s_ph2, 'ph_3': s_ph3
        })
# Capacitors as negative reactive loads (Y-connected, treated as type "Z" constant impedance load)
if ST.state.input_capacitors is not None:
    for _, cap in ST.state.input_capacitors.iterrows():
        bus_id = cap['bus']
        if bus_id in wb['id'].values:
            loads_list.append({
                'bus': bus_id,
                'conn': "Y",
                'type': "Z",
                'ph_1': -cap['kvar_ph1'] * 1000j,
                'ph_2': -cap['kvar_ph2'] * 1000j,
                'ph_3': -cap['kvar_ph3'] * 1000j
            })

# Normalize Type -> (np.int63(#) = #
for d in loads_list:
    d['bus'] = int(d['bus'])

# 11. Distributed generation: add as negative loads and populate generation_register
if ST.state.has_distributed_gen and ST.state.distributed_gen is not None:
    # PQ mode distributed generators
    pq_gen = ST.state.distributed_gen[ST.state.distributed_gen['mode'].str.upper() == "PQ"].copy()
    pq_gen.dropna(subset=['kw_set', 'kvar_set'], inplace=True)    # remove entries missing P or Q setpoints
    pq_gen = pq_gen[pq_gen['bus'].isin(wb['id'])]
    if not pq_gen.empty:
        for _, dg in pq_gen.iterrows():
            s_phase = (dg['kw_set'] + 1j*dg['kvar_set']) * 1000/3  # per-phase generation in VA
            loads_list.append({
                'bus': dg['bus'], 'conn': dg['conn'], 'type': dg['mode'],
                'ph_1': -s_phase, 'ph_2': -s_phase, 'ph_3': -s_phase
            })
            # Record this generator in generation_register (per-phase kW/kVAr in kW units, since divided by 3)
            ST.state.generation_register = pd.concat([
                ST.state.generation_register,
                pd.DataFrame([{
                    'bus': dg['bus'], 'mode': dg['mode'], 'conn': dg['conn'],
                    'kw_ph1': dg['kw_set'] / 3, 'kvar_ph1': dg['kvar_set'] / 3,
                    'kw_ph2': dg['kw_set'] / 3, 'kvar_ph2': dg['kvar_set'] / 3,
                    'kw_ph3': dg['kw_set'] / 3, 'kvar_ph3': dg['kvar_set'] / 3,
                    'max_diff': 0.0
                }])
            ], ignore_index=True)
        ST.state.has_pq_distributed_gen = True
        ST.state.pq_distributed_gen = pq_gen

    # PQV mode distributed generators
    pqv_gen = ST.state.distributed_gen[ST.state.distributed_gen['mode'].str.upper() == "PQV"].copy()
    pqv_gen.dropna(subset=['kw_set', 'kv_set', 'kvar_min', 'kvar_max', 'xd'], inplace=True)
    pqv_gen = pqv_gen[pqv_gen['bus'].isin(wb['id'])]
    if not pqv_gen.empty:
        for _, dg in pqv_gen.iterrows():
            p_phase = dg['kw_set'] * 1000/3
            q_phase = (dg['kvar_min'] + dg['kvar_max']) * 1000/6  # initial VAr = avg of min and max per phase
            s_phase = p_phase + 1j * q_phase
            loads_list.append({
                'bus': dg['bus'], 'conn': dg['conn'], 'type': dg['mode'],
                'ph_1': -s_phase, 'ph_2': -s_phase, 'ph_3': -s_phase
            })
            ST.state.generation_register = pd.concat([
                ST.state.generation_register,
                pd.DataFrame([{
                    'bus': dg['bus'], 'mode': dg['mode'], 'conn': dg['conn'],
                    'kw_ph1': dg['kw_set'] / 3,
                    'kvar_ph1': (dg['kvar_min'] + dg['kvar_max']) / 3,
                    'kw_ph2': dg['kw_set'] / 3,
                    'kvar_ph2': (dg['kvar_min'] + dg['kvar_max']) / 3,
                    'kw_ph3': dg['kw_set'] / 3,
                    'kvar_ph3': (dg['kvar_min'] + dg['kvar_max']) / 3,
                    'max_diff': 0.0
                }])
            ], ignore_index=True)
        # Prepare the PQV DG DataFrame for power flow (add extra columns for voltages and outputs)
        pqv_gen = pqv_gen[['bus','conn','mode','kw_set','kv_set','kvar_min','kvar_max','xd']].copy()
        pqv_gen[['v_ph1','v_ph2','v_ph3','max_diff']] = 0.0
        pqv_gen[['w_ph1','w_ph2','w_ph3','var_ph1','var_ph2','var_ph3']] = 0.0
        ST.state.has_pqv_distributed_gen = True
        ST.state.pqv_distributed_gen = pqv_gen

    # PI mode distributed generators
    pi_gen = ST.state.distributed_gen[ST.state.distributed_gen['mode'].str.upper() == "PI"].copy()
    pi_gen.dropna(subset=['kw_set', 'amp_set', 'kvar_min', 'kvar_max'], inplace=True)
    pi_gen = pi_gen[pi_gen['bus'].isin(wb['id'])]
    if not pi_gen.empty:
        for _, dg in pi_gen.iterrows():
            p_phase = dg['kw_set'] * 1000/3
            q_phase = (dg['kvar_min'] + dg['kvar_max']) * 1000/6
            s_phase = p_phase + 1j * q_phase
            loads_list.append({
                'bus': dg['bus'], 'conn': dg['conn'], 'type': dg['mode'],
                'ph_1': -s_phase, 'ph_2': -s_phase, 'ph_3': -s_phase
            })
            ST.state.generation_register = pd.concat([
                ST.state.generation_register,
                pd.DataFrame([{
                    'bus': dg['bus'], 'mode': dg['mode'], 'conn': dg['conn'],
                    'kw_ph1': dg['kw_set'] / 3,
                    'kvar_ph1': (dg['kvar_min'] + dg['kvar_max']) / 3,
                    'kw_ph2': dg['kw_set'] / 3,
                    'kvar_ph2': (dg['kvar_min'] + dg['kvar_max']) / 3,
                    'kw_ph3': dg['kw_set'] / 3,
                    'kvar_ph3': (dg['kvar_min'] + dg['kvar_max']) / 3,
                    'max_diff': 0.0
                }])
            ], ignore_index=True)
        # Prepare the PI DG DataFrame for power flow (add extra columns similar to PQV)
        pi_gen = pi_gen[['bus','conn','mode','kw_set','amp_set','kvar_min','kvar_max']].copy()
        pi_gen[['v_ph1','v_ph2','v_ph3','max_diff']] = 0.0
        pi_gen[['w_ph1','w_ph2','w_ph3','var_ph1','var_ph2','var_ph3']] = 0.0
        ST.state.has_pi_distributed_gen = True
        ST.state.pi_distributed_gen = pi_gen

    # If no valid DG entries remain, update the master flag
    if not (ST.state.has_pq_distributed_gen or ST.state.has_pqv_distributed_gen or ST.state.has_pi_distributed_gen):
        ST.state.has_distributed_gen = False

# 12. Create loads DataFrame from the assembled list
loads_df = pd.DataFrame(loads_list, columns=['bus','conn','type','ph_1','ph_2','ph_3']) if loads_list else \
           pd.DataFrame(columns=['bus','conn','type','ph_1','ph_2','ph_3'])
# 13. Add k_1, k_2, k_3 columns for load model constants (for constant Z and I loads)
loads_df[['k_1','k_2','k_3']] = None
# Calculate constants for constant impedance (Z) and constant current (I) loads
for idx, load in loads_df.iterrows():
    if load['type'] in ["Z", "I"]:
        bus_id = load['bus']
        # Base voltage magnitude at this bus (line-to-neutral)
        v_base = wb.loc[wb['id'] == bus_id, 'v_base'].values
        v_nom = v_base[0] if v_base.size > 0 else eln  # use bus v_base if set, else system base
        if load['conn'] == "D":
            v_nom *= math.sqrt(3)  # delta connection uses line-to-line voltage as base
        if load['type'] == "Z":
            # Constant impedance load: k = (V_nom^2 / |S|) * e^(j angle(S)) for each phase
            for phase in ['ph_1','ph_2','ph_3']:
                S = load[phase]
                if S == 0 or S == 0+0j:
                    loads_df.at[idx, 'k_'+phase[-1]] = 0+0j
                else:
                    mag_S = abs(S)
                    ang_S = np.angle(S) # Radian
                    loads_df.at[idx, 'k_'+phase[-1]] = (v_nom**2 / mag_S) * np.exp(1j * ang_S)
        else:
            # Constant current load: k = |S| / V_nom for each phase (phase angle irrelevant for magnitude)
            for phase in ['ph_1','ph_2','ph_3']:
                S = load[phase]
                if S == 0 or S == 0+0j:
                    loads_df.at[idx, 'k_'+phase[-1]] = 0+0j
                else:
                    loads_df.at[idx, 'k_'+phase[-1]] = abs(S) / v_nom
ST.state.loads = loads_df.copy()

# 14. Initialize bus processing flag and voltage/current columns for iteration
wb['process'] = 0
wb['phases'] = None
wb[['v_ph1','v_ph2','v_ph3']] = None
# Sort buses by number in descending order (for backward sweep processing order)
wb.sort_values(by='number', ascending=False, inplace=True)
wb.reset_index(drop=True, inplace=True)
# Initialize bus and line current accumulator columns to 0
wb[['ibus_1','ibus_2','ibus_3']] = 0+0j
lines[['ibus1_1','ibus1_2','ibus1_3']] = 0+0j
ST.state.working_buses = wb.copy()
ST.state.lines = lines.copy()

# Initialize temporary 3-phase voltage and current vectors for iteration sweeps
ST.state.Vbus1 = np.zeros((3, 1), dtype=complex)
ST.state.Vbus2 = np.zeros((3, 1), dtype=complex)
ST.state.Ibus1 = np.zeros((3, 1), dtype=complex)
ST.state.Ibus2 = np.zeros((3, 1), dtype=complex)

# Initialize sequence component placeholders (optional, for output)
ST.state.x_seq = None
ST.state.x_seq_df = pd.DataFrame()

# At this point, err_msg is an empty string if everything succeeded, or contains an error message if a failure occurred.
