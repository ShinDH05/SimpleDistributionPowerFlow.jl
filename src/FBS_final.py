import sys, math, os, csv
from datetime import datetime
import numpy as np

# ===== User-specified directories =====
input_dir  = r"C:\dev\ACPF\BFS\examples\ieee-13"
output_dir = r"C:\dev\ACPF\BFS\results"

# ==== User-adjustable parameters ====
CALLER = "powerflow"
TIMESTAMP = True            # append timestamp to output filenames
VERBOSE = 1

# ===== Global state variables =====
distributed_gen = []
distributed_loads = []
input_capacitors = []
input_segments = []
line_configurations = []
regulators = []
spot_loads = []
substation = []
switches = []
transformers = []
has_capacitor = False
has_distributed_gen = False
has_distributed_load = False
has_regulator = False
has_switch = False
has_transformer = False

# ===== Accumulated error messages =====
err_msg = ""

# --------------------------------------------------------------
# Normalise input/output directories
# --------------------------------------------------------------
# input
if input_dir != "":
    if os.path.exists(input_dir):
        if not os.path.isabs(input_dir):
            input_dir = os.path.join(os.getcwd(), input_dir)
    else:
        err_msg = f"{input_dir} is not a valid directory"
else:
    input_dir = os.getcwd()

# output
if output_dir != "":
    if os.path.exists(output_dir):
        if not os.path.isabs(output_dir):
            output_dir = os.path.join(os.getcwd(), output_dir)
    else:
        # Create directory if missing path
        os.makedirs(output_dir, exist_ok=True)
        if not os.path.isabs(output_dir):
            output_dir = os.path.join(os.getcwd(), output_dir)
else:
    output_dir = os.getcwd()

if err_msg:
    print(err_msg)
    sys.exit(1)

# --------------------------------------------------------------
#  - Load CSVs (check existence and emptiness)
#  - Validate column names and order
#  - Validate units (ft, mi, m, km)
#  - Check for duplicates
#  - Normalise text to uppercase (config/state/mode/conn)
# --------------------------------------------------------------
accepted_units = ["ft", "mi", "m", "km"]

# -- Utility: CSV reading pattern without functions --
# Returns: rows (list of dict) and file_err ("no file"/"empty file"/"")

# substation.csv
# newline="" prevents blank rows from OS line ending differences, ensuring consistent CSV parsing
# encoding="utf-8-sig" uses UTF-8 and automatically removes BOM (Byte Order Mark) to prevent parsing errors
substation_path = os.path.join(input_dir, "substation.csv")
if os.path.isfile(substation_path):
    with open(substation_path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        hdr = list(reader.fieldnames or [])
        rows = list(reader)
    if len(rows) == 0:
        file_err = "empty file"
    else:
        file_err = ""
else:
    rows = []
    file_err = "no file"
substation = rows

if file_err == "no file":
    err_msg = f"there is not 'substation.csv' file in {input_dir}"
    print(err_msg); sys.exit(1)
elif file_err == "empty file":
    err_msg = "'substation.csv' file is empty"
    print(err_msg); sys.exit(1)
else:
    expected = ["bus","kva","kv"]
    if hdr != expected:
        err_msg = "check for column names in 'substation.csv' file"
        print(err_msg); sys.exit(1)

# line_segments.csv
segments_path = os.path.join(input_dir, "line_segments.csv")
if os.path.isfile(segments_path):
    with open(segments_path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        seg_hdr = list(reader.fieldnames or [])
        seg_rows = list(reader)
    if len(seg_rows) == 0:
        seg_err = "empty file"
    else:
        seg_err = ""
else:
    seg_rows = []
    seg_err = "no file"
input_segments = seg_rows

if seg_err == "no file":
    err_msg = f"there is not 'line_segments.csv' file in {input_dir}"
    print(err_msg); sys.exit(1)
elif seg_err == "empty file":
    err_msg = "'line_segments.csv' file is empty"
    print(err_msg); sys.exit(1)
else:
    expected = ["bus1","bus2","length","unit","config"]
    if seg_hdr != expected:
        err_msg = "check for column names in 'line_segments.csv' file"
        print(err_msg); sys.exit(1)
    # Validate units
    bad_units = [r for r in input_segments if (r.get("unit") not in accepted_units)]
    if len(bad_units) > 0:
        err_msg = "check for units in 'line_segments.csv' file (only ft, mi, m and km are accepted units)"
        print(err_msg); sys.exit(1)
    # Check for duplicate (bus1, bus2) pairs
    pair_set = set()
    for r in input_segments:
        pair = (r.get("bus1"), r.get("bus2"))
        pair_set.add(pair)
    if len(pair_set) != len(input_segments):
        err_msg = "check for duplicated links in 'line_segments.csv' file"
        print(err_msg); sys.exit(1)
    # Force configuration codes to uppercase strings
    for r in input_segments:
        r["config"] = str(r.get("config", "")).upper()

# line_configurations.csv
cfg_path = os.path.join(input_dir, "line_configurations.csv")
if os.path.isfile(cfg_path):
    with open(cfg_path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        cfg_hdr = list(reader.fieldnames or [])
        cfg_rows = list(reader)
    if len(cfg_rows) == 0:
        cfg_err = "empty file"
    else:
        cfg_err = ""
else:
    cfg_rows = []
    cfg_err = "no file"
line_configurations = cfg_rows

if cfg_err == "no file":
    err_msg = f"there is not 'line_configurations.csv' file in {input_dir}"
    print(err_msg); sys.exit(1)
elif cfg_err == "empty file":
    err_msg = "'line_configurations.csv' file is empty"
    print(err_msg); sys.exit(1)
else:
    expected = ["config","unit","raa","xaa","rab","xab","rac","xac","rbb","xbb","rbc","xbc","rcc","xcc","baa","bab","bac","bbb","bbc","bcc"]
    if cfg_hdr != expected:
        err_msg = "check for column names in 'line_configurations.csv' file"
        print(err_msg); sys.exit(1)
    # Validate units
    bad_units = [r for r in line_configurations if (r.get("unit") not in accepted_units)]
    if len(bad_units) > 0:
        err_msg = "check for units in 'line_configurations.csv' file (only ft, mi, m and km are accepted units)"
        print(err_msg); sys.exit(1)
    # Uppercase configuration codes and check for duplicates
    for r in line_configurations:
        r["config"] = str(r.get("config", "")).upper()
    cfg_codes = [r.get("config") for r in line_configurations]
    if len(set(cfg_codes)) != len(cfg_codes):
        err_msg = "check for duplicated configuration code in 'line_configurations.csv' file"
        print(err_msg); sys.exit(1)

# transformers.csv (optional)
tr_path = os.path.join(input_dir, "transformers.csv")
if os.path.isfile(tr_path):
    with open(tr_path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        tr_hdr = list(reader.fieldnames or [])
        tr_rows = list(reader)
    if len(tr_rows) == 0:
        tr_err = "empty file"
    else:
        tr_err = ""
else:
    tr_rows = []
    tr_err = "no file"
transformers = tr_rows

if tr_err == "":
    expected = ["config","kva","phases","conn_high","conn_low","kv_high","kv_low","rpu","xpu"]
    if tr_hdr != expected:
        err_msg = "check for column names in 'transformers.csv' file"
        print(err_msg); sys.exit(1)
    else:
        for r in transformers:
            r["config"] = str(r.get("config", "")).upper()
            r["conn_high"] = str(r.get("conn_high", "")).upper()
            r["conn_low"]  = str(r.get("conn_low", "")).upper()
        tr_cfg = [r.get("config") for r in transformers]
        if len(set(tr_cfg)) != len(tr_cfg):
            err_msg = "check for duplicated configuration code in 'transformers.csv' file"
            print(err_msg); sys.exit(1)
        has_transformer = True

# switches.csv (optional)
sw_path = os.path.join(input_dir, "switches.csv")
if os.path.isfile(sw_path):
    with open(sw_path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        sw_hdr = list(reader.fieldnames or [])
        sw_rows = list(reader)
    if len(sw_rows) == 0:
        sw_err = "empty file"
    else:
        sw_err = ""
else:
    sw_rows = []
    sw_err = "no file"
switches = sw_rows

if sw_err == "":
    expected = ["config","phases","state","resistance"]
    if sw_hdr != expected:
        err_msg = "check for column names in 'switches.csv' file"
        print(err_msg); sys.exit(1)
    else:
        for r in switches:
            r["config"] = str(r.get("config", "")).upper()
            r["state"]  = str(r.get("state", "")).upper()
        sw_cfg = [r.get("config") for r in switches]
        if len(set(sw_cfg)) != len(sw_cfg):
            err_msg = "check for duplicated configuration code in 'switches.csv' file"
            print(err_msg); sys.exit(1)
        has_switch = True

# regulators.csv (optional)
rg_path = os.path.join(input_dir, "regulators.csv")
if os.path.isfile(rg_path):
    with open(rg_path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rg_hdr = list(reader.fieldnames or [])
        rg_rows = list(reader)
    if len(rg_rows) == 0:
        rg_err = "empty file"
    else:
        rg_err = ""
else:
    rg_rows = []
    rg_err = "no file"
regulators = rg_rows

if rg_err == "":
    expected = ["config","phases","mode","tap_1","tap_2","tap_3"]
    if rg_hdr != expected:
        err_msg = "check for column names in 'regulators.csv' file."
        print(err_msg); sys.exit(1)
    else:
        for r in regulators:
            r["config"] = str(r.get("config", "")).upper()
            r["mode"]   = str(r.get("mode", "")).upper()
        rg_cfg = [r.get("config") for r in regulators]
        if len(set(rg_cfg)) != len(rg_cfg):
            err_msg = "check for duplicated configuration code in 'regulators.csv' file"
            print(err_msg); sys.exit(1)
        has_regulator = True

# Validate segment configurations (line/transformer/switch/regulator)
seg_cfgs = [r.get("config") for r in input_segments]
cfg_line_set = set([r.get("config") for r in line_configurations])
without_config = [r for r in input_segments if r.get("config") not in cfg_line_set]

if len(without_config) != 0 and has_transformer:
    tr_cfg_set = set([r.get("config") for r in transformers])
    without_config = [r for r in without_config if r.get("config") not in tr_cfg_set]

if len(without_config) != 0 and has_switch:
    sw_cfg_set = set([r.get("config") for r in switches])
    without_config = [r for r in without_config if r.get("config") not in sw_cfg_set]

if len(without_config) != 0 and has_regulator:
    rg_cfg_set = set([r.get("config") for r in regulators])
    without_config = [r for r in without_config if r.get("config") not in rg_cfg_set]

if len(without_config) != 0:
    missing_codes = [r.get("config") for r in without_config]
    err_msg = (f"check for {missing_codes} code(s) in 'line_segments', 'line_configurations', "
               f"'transformers', 'switches' or 'regulators' .csv files in {input_dir}")
    print(err_msg); sys.exit(1)

# distributed_loads.csv (optional)
dl_path = os.path.join(input_dir, "distributed_loads.csv")
if os.path.isfile(dl_path):
    with open(dl_path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        dl_hdr = list(reader.fieldnames or [])
        dl_rows = list(reader)
    if len(dl_rows) == 0:
        dl_err = "empty file"
    else:
        dl_err = ""
else:
    dl_rows = []
    dl_err = "no file"
distributed_loads = dl_rows

if dl_err == "":
    expected = ["bus1","bus2","conn","type","kw_ph1","kvar_ph1","kw_ph2","kvar_ph2","kw_ph3","kvar_ph3"]
    if dl_hdr != expected:
        err_msg = "check for column names in 'distributed_loads.csv' file"
        print(err_msg); sys.exit(1)
    else:
        has_distributed_load = True
else:
    if VERBOSE != 0:
        print("no distributed loads")
    # Julia clears err_msg here
    dl_err = ""

# Additional inputs when CALLER is "powerflow"
if CALLER == "powerflow":
    # spot_loads.csv (required)
    sl_path = os.path.join(input_dir, "spot_loads.csv")
    if os.path.isfile(sl_path):
        with open(sl_path, "r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            sl_hdr = list(reader.fieldnames or [])
            sl_rows = list(reader)
        if len(sl_rows) == 0:
            sl_err = "empty file"
        else:
            sl_err = ""
    else:
        sl_rows = []
        sl_err = "no file"
    spot_loads = sl_rows

    if sl_err == "no file":
        err_msg = f"there is not 'spot_loads.csv' file in {input_dir}"
        print(err_msg); sys.exit(1)
    elif sl_err == "empty file":
        err_msg = "'spot_loads.csv' file is empty"
        print(err_msg); sys.exit(1)
    else:
        expected = ["bus","conn","type","kw_ph1","kvar_ph1","kw_ph2","kvar_ph2","kw_ph3","kvar_ph3"]
        if sl_hdr != expected:
            err_msg = "check for column names in 'spot_loads.csv' file"
            print(err_msg); sys.exit(1)

    # capacitors.csv (optional)
    cap_path = os.path.join(input_dir, "capacitors.csv")
    if os.path.isfile(cap_path):
        with open(cap_path, "r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            cap_hdr = list(reader.fieldnames or [])
            cap_rows = list(reader)
        if len(cap_rows) == 0:
            cap_err = "empty file"
        else:
            cap_err = ""
    else:
        cap_rows = []
        cap_err = "no file"
    input_capacitors = cap_rows

    if cap_err == "":
        expected = ["bus","kvar_ph1","kvar_ph2","kvar_ph3"]
        if cap_hdr != expected:
            err_msg = "check for column names in 'capacitors.csv' file"
            print(err_msg); sys.exit(1)
        else:
            has_capacitor = True
    else:
        if VERBOSE != 0:
            print("no capacitors")
        cap_err = ""

    # distributed_generation.csv (optional)
    dg_path = os.path.join(input_dir, "distributed_generation.csv")
    if os.path.isfile(dg_path):
        with open(dg_path, "r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            dg_hdr = list(reader.fieldnames or [])
            dg_rows = list(reader)
        if len(dg_rows) == 0:
            dg_err = "empty file"
        else:
            dg_err = ""
    else:
        dg_rows = []
        dg_err = "no file"
    distributed_gen = dg_rows

    if dg_err == "":
        expected = ["bus","conn","mode","kw_set","kvar_set","kv_set","amp_set","kvar_min","kvar_max","xd"]
        if dg_hdr != expected:
            err_msg = "check for column names in 'distributed_generation.csv' file"
            print(err_msg); sys.exit(1)
        else:
            # Normalize mode to uppercase and validate
            for r in distributed_gen:
                r["mode"] = str(r.get("mode", "")).upper()
            invalid_modes = [r for r in distributed_gen if r.get("mode") not in ("PQ", "PQV", "PI")]
            if len(invalid_modes) > 0:
                err_msg = ("modes of Distributed Generation accepted: PQ (traditional constant watt-var), "
                           "PQV (volt dependant var PQ), PI (constant watt-ampere)")
                print(err_msg); sys.exit(1)
            # Report and drop rows with missing bus/conn/mode (Julia `dropmissing!`)
            def _is_missing(v):
                return v is None or (isinstance(v, str) and v.strip() == "")
            before = len(distributed_gen)
            distributed_gen = [r for r in distributed_gen if not (_is_missing(r.get("bus")) or
                                                                 _is_missing(r.get("conn")) or
                                                                 _is_missing(r.get("mode")))]
            after = len(distributed_gen)
            if before != after:
                print("Distributed generation registers with missing values will be ignored.")
            has_distributed_gen = (len(distributed_gen) != 0)
            if not has_distributed_gen:
                if VERBOSE != 0:
                    print("no distributed generation")

    else:
        if VERBOSE != 0:
            print("no distributed generation")

# Final: exit normally if no errors
# Subsequent stages reuse these global states
if VERBOSE != 0:
    print("\nInput files loaded and validated successfully.")

# === topology_discovery ===
# Prerequisite: the previous block prepared these globals:
#   - Input data: substation, input_segments, line_configurations, transformers,
#     switches, regulators, distributed_loads
#   - Flags: has_transformer, has_switch, has_regulator, has_distributed_load,
#   - Paths: input_dir, output_dir
# Note: graph visualisation is omitted; results are written to CSV instead.

# ==== 0) Build bus list for input topology (original: input_buses) ====
# Rule: unique bus2 order from line_segments, then add any remaining bus1 values
_input_bus_ids = []
for seg in input_segments:
    b2 = int(seg["bus2"])
    if b2 not in _input_bus_ids:
        _input_bus_ids.append(b2)
for seg in input_segments:
    b1 = int(seg["bus1"])
    if b1 not in _input_bus_ids:
        _input_bus_ids.append(b1)

# Bus table (original uses DataFrame with only the :id column)
input_buses = [{"id": b} for b in _input_bus_ids]

# ==== 1) Adjacency matrix for input topology (with switch states) ====
# Replicates original adjacency_matrix(buses, segments) logic
# - buses sorted for indexing
_ids_sorted = sorted([int(r["id"]) for r in input_buses])
_id2idx = {bid: i for i, bid in enumerate(_ids_sorted)}
N_in = len(_ids_sorted)
adj_mat_input = np.zeros((N_in, N_in), dtype=np.int64)

# Base edges (direction: bus1 -> bus2)
for seg in input_segments:
    i = _id2idx[int(seg["bus1"])]
    j = _id2idx[int(seg["bus2"])]
    adj_mat_input[i, j] = 1

# Remove edges for switch segments based on state
if has_switch:
    line_cfg_codes = set([str(r["config"]) for r in line_configurations])
    sw_cfg_codes = set([str(r["config"]) for r in switches])
    for seg in input_segments:
        cfg = str(seg["config"])
        if (cfg not in line_cfg_codes) and (cfg in sw_cfg_codes):
            # This segment represents a switch
            # Check the switch state
            st = None
            for s in switches:
                if str(s["config"]) == cfg:
                    st = str(s["state"]).upper()
                    break
            if st is not None and st != "CLOSED":
                i = _id2idx[int(seg["bus1"])]
                j = _id2idx[int(seg["bus2"])]
                adj_mat_input[i, j] = 0

# ==== 2) Build working topology (apply switches, remove isolated segments) ====
# Start with input_segments plus a `check` column; mark open switches with check=1
working_segments = []
for seg in input_segments:
    rec = dict(seg)
    rec["_check"] = 0
    working_segments.append(rec)

if has_switch:
    for m in range(len(working_segments)):
        seg = working_segments[m]
        for s in switches:
            if str(seg["config"]) == str(s["config"]):
                if str(s["state"]).upper() != "CLOSED":
                    working_segments[m]["_check"] = 1
    working_segments = [r for r in working_segments if r["_check"] == 0]

# ==== 3) Keep only buses reachable from the root ====
# Seed: substation bus (:bus → :id)
if not substation:
    raise RuntimeError("substation.csv is required and must have at least one row.")
root_bus = int(substation[0]["bus"])
working_buses_ids = [root_bus]
increase_monitor = 1

# Same as original: expand working_buses sequentially and stop when no new bus is found
for _ in range(len(working_segments)):
    # Search edges starting from the bus at current index n
    n_idx = len(working_buses_ids) - 1  # index of the last added bus
    # Original iterates 1..nrow(working_segments) using working_buses[n,:id]
    # Here we expand from all collected buses to achieve the same effect
    before = len(working_buses_ids)
    for known in list(working_buses_ids):
        for seg in working_segments:
            if int(seg["bus1"]) == known:
                b2 = int(seg["bus2"])
                if b2 not in working_buses_ids:
                    working_buses_ids.append(b2)
    if len(working_buses_ids) > increase_monitor:
        increase_monitor += 1
    else:
        break

# ==== 4) Correct direction due to switches (swap segments if bus missing) ====
if has_switch and (len(working_buses_ids) != len(_input_bus_ids)):
    buses_diff = len(_input_bus_ids) - len(working_buses_ids)
    for _ in range(buses_diff):
        # Candidate segments where bus2 is connected but bus1 is missing
        tmp = []
        for seg in working_segments:
            b1 = int(seg["bus1"]); b2 = int(seg["bus2"])
            if (b2 in working_buses_ids) and (b1 not in working_buses_ids):
                tmp.append({"bus1": b1, "bus2": b2})
        if len(tmp) > 0:
            pick = tmp[0]
            # Add missing bus
            working_buses_ids.append(int(pick["bus1"]))
            # Swap segment direction accordingly
            for seg in working_segments:
                if int(seg["bus1"]) == pick["bus1"] and int(seg["bus2"]) == pick["bus2"]:
                    seg["bus1"], seg["bus2"] = seg["bus2"], seg["bus1"]

# ==== 5) Remove working segments unreachable from the root ====
for seg in working_segments:
    if int(seg["bus1"]) not in working_buses_ids:
        seg["_check"] = 1
working_segments = [r for r in working_segments if r["_check"] == 0]
for seg in working_segments:
    if "_check" in seg: del seg["_check"]

# ==== 6) Check for loops ====
if (len(working_segments) - len(working_buses_ids) + 1) > 0:
    msg = f"Topology has a loop, this version only works with radial topologies. See result in {output_dir}."
    print(msg)
    raise RuntimeError(msg)

# ==== 7) Handle distributed loads: insert auxiliary buses (split segments) ====
auxiliar_buses = []   # [{bus1, bus2, busx}]
if has_distributed_load and distributed_loads:
    # Convert length to float
    for seg in working_segments:
        seg["length"] = float(seg["length"])

    # Collect segments corresponding to distributed loads
    dist_load_segments = []
    for dl in distributed_loads:
        b1 = int(dl["bus1"]); b2 = int(dl["bus2"])
        for seg in working_segments:
            if int(seg["bus1"]) == b1 and int(seg["bus2"]) == b2:
                dist_load_segments.append(dict(seg))  # copy

    # Remove collected segments from original working segments
    def _is_same(a, b):
        return int(a["bus1"]) == int(b["bus1"]) and int(a["bus2"]) == int(b["bus2"])
    working_segments = [s for s in working_segments if all(not _is_same(s, d) for d in dist_load_segments)]

    # Insert intermediate bus and split into two segments
    next_bus_id = max(working_buses_ids) + 1 if working_buses_ids else 1
    for dseg in dist_load_segments:
        start_bus = int(dseg["bus1"])
        end_bus   = int(dseg["bus2"])
        unit = dseg["unit"]; conf = dseg["config"]
        L = float(dseg["length"]); L1 = L * 0.5; L2 = L * 0.5
        # first half
        working_segments.append({"bus1": start_bus, "bus2": next_bus_id,
                                 "length": L1, "unit": unit, "config": conf})
        # second half
        working_segments.append({"bus1": next_bus_id, "bus2": end_bus,
                                 "length": L2, "unit": unit, "config": conf})
        auxiliar_buses.append({"bus1": start_bus, "bus2": end_bus, "busx": next_bus_id})
        working_buses_ids.append(next_bus_id)
        next_bus_id += 1

# ==== 8) Compute adjacency matrix for working topology ====
_ids_sorted_w = sorted(list(set(int(b) for b in working_buses_ids)))
_id2idx_w = {bid: i for i, bid in enumerate(_ids_sorted_w)}
N_w = len(_ids_sorted_w)
adj_mat_work = np.zeros((N_w, N_w), dtype=np.int64)

for seg in working_segments:
    i = _id2idx_w[int(seg["bus1"])]
    j = _id2idx_w[int(seg["bus2"])]
    adj_mat_work[i, j] = 1

# Switch segments (devices) set to 0 when open
if has_switch:
    line_cfg_codes = set([str(r["config"]) for r in line_configurations])
    sw_cfg_codes = set([str(r["config"]) for r in switches])
    for seg in working_segments:
        cfg = str(seg["config"])
        if (cfg not in line_cfg_codes) and (cfg in sw_cfg_codes):
            st = None
            for s in switches:
                if str(s["config"]) == cfg:
                    st = str(s["state"]).upper()
                    break
            if st is not None and st != "CLOSED":
                i = _id2idx_w[int(seg["bus1"])]
                j = _id2idx_w[int(seg["bus2"])]
                adj_mat_work[i, j] = 0

# === data_preparation ===
# Prerequisite: previous stages prepared the following globals:
#  - substation, working_segments, line_configurations, transformers, switches,
#    regulators, spot_loads, distributed_loads, input_capacitors, distributed_gen
#  - has_transformer, has_switch, has_regulator, has_distributed_load,
#    has_distributed_gen
#  - input_dir, output_dir

# ---------- Common utilities ----------
def _to_float(x):
    try:
        return float(x)
    except Exception:
        return float('nan')

def _to_int(x):
    try:
        return int(x)
    except Exception:
        return 0

def _is_missing(v):
    return v is None or (isinstance(v, str) and v.strip() == "")

def _rows_where(seq, pred):
    return [r for r in seq if pred(r)]

def _first_row(seq, pred):
    for r in seq:
        if pred(r):
            return r
    return None

def _ids_from_segments(segs):
    s = set()
    for r in segs:
        s.add(_to_int(r["bus1"]))
        s.add(_to_int(r["bus2"]))
    return sorted(s)

# ---------- Ensure working_buses / adj_mat are available ----------
try:
    working_buses
except NameError:
    working_buses = []

if not working_buses:
    # If topology stage didn't build the list, derive it from segments
    _wb_ids = _ids_from_segments(working_segments)
    working_buses = [{"id": i} for i in _wb_ids]

try:
    adj_mat  # check if created during topology stage
except NameError:
    # If missing, build from working_segments
    _wb_ids = sorted([b["id"] for b in working_buses])
    _idx = {bid: i for i, bid in enumerate(_wb_ids)}
    N = len(_wb_ids)
    adj_mat = np.zeros((N, N), dtype=np.int64)
    for seg in working_segments:
        i = _idx[_to_int(seg["bus1"])]
        j = _idx[_to_int(seg["bus2"])]
        adj_mat[i, j] = 1

# ---------- working_lines() conversion block ----------
# lines = working_segments + :type (1=line, 2=transformer, 3=switch, 4=regulator)
lines = [dict(r) for r in working_segments]
for r in lines:
    r["type"] = 0  # initial value

# Determine type
line_cfg_codes = set(str(r["config"]) for r in line_configurations)
if has_transformer:
    tr_codes = set(str(r["config"]) for r in transformers)
if has_switch:
    sw_codes = set(str(r["config"]) for r in switches)
if has_regulator:
    rg_codes = set(str(r["config"]) for r in regulators)

for r in lines:
    cfg = str(r["config"])
    if cfg in line_cfg_codes:
        r["type"] = 1
    if has_transformer and cfg in tr_codes:
        r["type"] = 2
    if has_switch and cfg in sw_codes:
        r["type"] = 3
    if has_regulator and cfg in rg_codes:
        r["type"] = 4

# Add :type and :number columns to working_buses
for b in working_buses:
    b["type"] = 0   # 1=sub,2=bif,3=intermediate,4=next-to-end,5=end
    b["number"] = 0

# Compute out-degree (=downward_buses)
downward_buses = np.sum(adj_mat, axis=1).astype(np.int64)  # shape (N,)
# Map bus id to row index
wb_ids_sorted = sorted([b["id"] for b in working_buses])
wb_id2idx = {bid: i for i, bid in enumerate(wb_ids_sorted)}

# First marking: end vs branch
for b in working_buses:
    deg = downward_buses[wb_id2idx[b["id"]]]
    if deg == 0:
        b["type"] = 5  # ending
    elif deg > 1:
        b["type"] = 2  # bifurcation

# Second marking: if out-degree=1 and next bus is an end bus, mark 4 else 3
for b in working_buses:
    if downward_buses[wb_id2idx[b["id"]]] == 1:
        # b→next_bus
        seg = _first_row(working_segments, lambda s: _to_int(s["bus1"]) == b["id"])
        if seg is not None:
            nxt = _first_row(working_buses, lambda x: x["id"] == _to_int(seg["bus2"]))
            if nxt is not None and nxt["type"] == 5:
                b["type"] = 4  # next-to-end
            else:
                b["type"] = 3  # intermediate

# Mark substation bus
root_bus = _to_int(substation[0]["bus"])
for b in working_buses:
    if b["id"] == root_bus:
        b["type"] = 1

# Sort by type ascending
working_buses.sort(key=lambda x: x["type"])

# Assign initial numbers (type ∈ {1,4,5})
for idx, b in enumerate(working_buses, start=1):
    if b["type"] in (1, 4, 5):
        b["number"] = idx

# Statistics and starting index k for reverse numbering
initial_buses     = sum(1 for b in working_buses if b["type"] == 1)
bifurcation_buses = sum(1 for b in working_buses if b["type"] == 2)
interm_buses      = sum(1 for b in working_buses if b["type"] == 3)
next_to_end_buses = sum(1 for b in working_buses if b["type"] == 4)
end_buses         = sum(1 for b in working_buses if b["type"] == 5)
k = len(working_buses) - end_buses - next_to_end_buses

# Number intermediate and branch buses
if bifurcation_buses > 0:
    for _p in range(bifurcation_buses):
        for _q in range(next_to_end_buses):
            for b in working_buses:
                if b["type"] == 3 and b["number"] == 0:
                    # b → seg(bus1=b.id) → n2
                    for seg in working_segments:
                        if _to_int(seg["bus1"]) == b["id"]:
                            n2 = _first_row(working_buses, lambda x: x["id"] == _to_int(seg["bus2"]))
                            if n2 and n2["number"] != 0:
                                b["number"] = k
                                k -= 1
        for b in working_buses:
            if b["type"] == 2 and b["number"] == 0:
                waiting = 0
                for seg in working_segments:
                    if _to_int(seg["bus1"]) == b["id"]:
                        n2 = _first_row(working_buses, lambda x: x["id"] == _to_int(seg["bus2"]))
                        if n2 and n2["number"] == 0:
                            waiting = 1
                if waiting == 0:
                    b["number"] = k
                    k -= 1
else:
    for b in working_buses:
        if b["type"] == 3:
            for seg in working_segments:
                if _to_int(seg["bus1"]) == b["id"]:
                    n2 = _first_row(working_buses, lambda x: x["id"] == _to_int(seg["bus2"]))
                    if n2 and n2["number"] != 0:
                        b["number"] = n2["number"] - 1

# Assign numbers to remaining buses (predecessor number + 1)
def _has_unnumbered():
    return any(b["type"] == 3 and b["number"] == 0 for b in working_buses)

while _has_unnumbered():
    for b in working_buses:
        if b["number"] == 0:
            prec_seg = _first_row(working_segments, lambda s: _to_int(s["bus2"]) == b["id"])
            if prec_seg:
                prec_bus = _first_row(working_buses, lambda x: x["id"] == _to_int(prec_seg["bus1"]))
                if prec_bus and prec_bus["number"] > 0:
                    b["number"] = prec_bus["number"] + 1
# Sort by number ascending
working_buses.sort(key=lambda x: x["number"])

# Mark downstream transformer buses
for b in working_buses:
    b["trf"] = None

if has_transformer and transformers:
    for t in transformers:
        for seg in working_segments:
            if str(seg["config"]) == str(t["config"]):
                child = _first_row(working_buses, lambda x: x["id"] == _to_int(seg["bus2"]))
                if child:
                    child["trf"] = t["config"]

    # propagate
    for b in working_buses:
        if (b["trf"] is not None) and (b["type"] != 5):
            for seg in working_segments:
                if _to_int(seg["bus1"]) == b["id"]:
                    child = _first_row(working_buses, lambda x: x["id"] == _to_int(seg["bus2"]))
                    if child:
                        child["trf"] = b["trf"]

# ---------- data_preparation() conversion block ----------
err_msg = ""

# Rotation constant and transformation matrix
as_ = np.exp(1j * np.deg2rad(120.0))            # = 1 * exp(j*120°)
As  = np.array([[1, 1, 1],
                [1, as_**2, as_],
                [1, as_, as_**2]], dtype=np.complex128)  # note: As unused later

# Base voltages (LL/LN) and transformation matrix D
ell = _to_float(substation[0]["kv"]) * 1000.0                    # [V_LL]
eln = ell / math.sqrt(3.0)                                       # [V_LN]
ELN = np.array([eln,
                eln * np.exp(-1j * np.deg2rad(120.0)),
                eln * np.exp( 1j * np.deg2rad(120.0))], dtype=np.complex128)
D = np.array([[ 1.0, -1.0,  0.0],
              [ 0.0,  1.0, -1.0],
              [-1.0,  0.0,  1.0]], dtype=np.float64)
ELL = D @ ELN.astype(np.complex128)

# Add v_base to working_buses (kv_low/√3 for transformer secondary, else ELN)
for b in working_buses:
    b["v_base"] = 0.0
for b in working_buses:
    if b.get("trf") is not None and has_transformer:
        for t in transformers:
            if str(b["trf"]) == str(t["config"]):
                b["v_base"] = _to_float(t["kv_low"]) * 1000.0 / math.sqrt(3.0)
                break
    else:
        b["v_base"] = eln

# Build line_configs (R/X → complex impedance, B coefficients j*value)
def _cfg_row(cfg):
    return {
        "config": str(cfg["config"]),
        "unit":   str(cfg["unit"]),
        "zaa": (_to_float(cfg["raa"]) + 1j * _to_float(cfg["xaa"])),
        "zab": (_to_float(cfg["rab"]) + 1j * _to_float(cfg["xab"])),
        "zac": (_to_float(cfg["rac"]) + 1j * _to_float(cfg["xac"])),
        "zbb": (_to_float(cfg["rbb"]) + 1j * _to_float(cfg["xbb"])),
        "zbc": (_to_float(cfg["rbc"]) + 1j * _to_float(cfg["xbc"])),
        "zcc": (_to_float(cfg["rcc"]) + 1j * _to_float(cfg["xcc"])),
        # baa..bcc multiplied by 1j in original, then by length*unit_factor*(1e-6)
        "baa": 1j * _to_float(cfg["baa"]),
        "bab": 1j * _to_float(cfg["bab"]),
        "bac": 1j * _to_float(cfg["bac"]),
        "bbb": 1j * _to_float(cfg["bbb"]),
        "bbc": 1j * _to_float(cfg["bbc"]),
        "bcc": 1j * _to_float(cfg["bcc"]),
    }

line_configs = [_cfg_row(r) for r in line_configurations]

# Add phases/Zxx/Bxx columns to lines (initially None)
for r in lines:
    r["phases"] = None
    for key in ("Zaa","Zab","Zac","Zbb","Zbc","Zcc","Baa","Bab","Bac","Bbb","Bbc","Bcc"):
        r[key] = None

# Append transformer impedance Zt (Ω)
if has_transformer and transformers:
    for t in transformers:
        kv_low = _to_float(t["kv_low"])
        kva    = _to_float(t["kva"])
        zpu    = _to_float(t["rpu"]) + 1j * _to_float(t["xpu"])
        t["Zt"] = (kv_low**2 / kva) * zpu * 1000.0  # [Ω], complex128

# Determine per-line Z/Y and phases
for r in lines:
    t = _to_int(r["type"])
    if t == 1:  # line
        cfg = _first_row(line_configs, lambda c: c["config"] == str(r["config"]))
        if cfg is None:
            continue
        # Unit conversion factor (segment unit → configuration unit)
        unit_line = str(r["unit"]).lower()
        unit_cfg = str(cfg["unit"]).lower()
        conv = {
            ("ft", "mi"): 1 / 5280.0, ("mi", "ft"): 5280.0,
            ("m", "km"): 1 / 1000.0, ("km", "m"): 1000.0,
            ("m", "mi"): 1 / 1609.344, ("mi", "m"): 1609.344,
            ("ft", "km"): 1 / 3280.8399, ("km", "ft"): 3280.8399
        }
        factor = conv.get((unit_line, unit_cfg), 1.0)
        L = _to_float(r["length"])
        r["Zaa"] = cfg["zaa"] * L * factor
        r["Zab"] = cfg["zab"] * L * factor
        r["Zac"] = cfg["zac"] * L * factor
        r["Zbb"] = cfg["zbb"] * L * factor
        r["Zbc"] = cfg["zbc"] * L * factor
        r["Zcc"] = cfg["zcc"] * L * factor
        # Convert µS to S by multiplying 1e-6; cfg["baa"] etc already j*value
        r["Baa"] = cfg["baa"] * L * factor * 1e-6
        r["Bab"] = cfg["bab"] * L * factor * 1e-6
        r["Bac"] = cfg["bac"] * L * factor * 1e-6
        r["Bbb"] = cfg["bbb"] * L * factor * 1e-6
        r["Bbc"] = cfg["bbc"] * L * factor * 1e-6
        r["Bcc"] = cfg["bcc"] * L * factor * 1e-6
        # Determine phases
        Zaa0 = (r["Zaa"] == 0 or r["Zaa"] == 0.0)
        Zbb0 = (r["Zbb"] == 0 or r["Zbb"] == 0.0)
        Zcc0 = (r["Zcc"] == 0 or r["Zcc"] == 0.0)
        if  Zaa0 and  Zbb0 and not Zcc0: r["phases"] = "c"
        if  Zaa0 and not Zbb0 and  Zcc0: r["phases"] = "b"
        if not Zaa0 and  Zbb0 and  Zcc0: r["phases"] = "a"
        if not Zaa0 and not Zbb0 and  Zcc0: r["phases"] = "ab"
        if not Zaa0 and  Zbb0 and not Zcc0: r["phases"] = "ac"
        if  Zaa0 and not Zbb0 and not Zcc0: r["phases"] = "bc"
        if not Zaa0 and not Zbb0 and not Zcc0: r["phases"] = "abc"

    elif t == 2 and has_transformer:
        tr = _first_row(transformers, lambda x: str(x["config"]) == str(r["config"]))
        if tr:
            Zt = tr["Zt"]
            r["Zaa"] = Zt; r["Zbb"] = Zt; r["Zcc"] = Zt
            r["Zab"] = 0;  r["Zac"] = 0;  r["Zbc"] = 0
            r["Baa"] = 0;  r["Bab"] = 0;  r["Bac"] = 0
            r["Bbb"] = 0;  r["Bbc"] = 0;  r["Bcc"] = 0
            r["phases"] = tr["phases"]

    elif t == 3 and has_switch:
        sw = _first_row(switches, lambda x: str(x["config"]) == str(r["config"]))
        if sw:
            if str(sw["state"]).upper() == "CLOSED":
                R = _to_float(sw["resistance"])
                r["Zaa"] = R; r["Zbb"] = R; r["Zcc"] = R
            else:
                r["Zaa"] = np.inf; r["Zbb"] = np.inf; r["Zcc"] = np.inf
            r["Zab"] = 0;  r["Zac"] = 0;  r["Zbc"] = 0
            r["Baa"] = 0;  r["Bab"] = 0;  r["Bac"] = 0
            r["Bbb"] = 0;  r["Bbc"] = 0;  r["Bcc"] = 0
            r["phases"] = sw["phases"]

    elif t == 4 and has_regulator:
        rg = _first_row(regulators, lambda x: str(x["config"]) == str(r["config"]))
        if rg:
            r["Zaa"]=r["Zab"]=r["Zac"]=r["Zbb"]=r["Zbc"]=r["Zcc"]=0
            r["Baa"]=r["Bab"]=r["Bac"]=r["Bbb"]=r["Bbc"]=r["Bcc"]=0
            r["phases"] = rg["phases"]

# Build generalized line matrix gen_lines_mat (Kersting 4ed ch.6)
gen_lines_mat = []
U = np.eye(3, dtype=np.complex128)
z_line = np.zeros((3,3), dtype=np.complex128)
y_line = np.zeros((3,3), dtype=np.complex128)

for r in lines:
    # Form symmetric matrices (Zxx, Bxx set above)
    z_line[0,0] = r["Zaa"]; z_line[0,1] = r["Zab"]; z_line[0,2] = r["Zac"]
    z_line[1,0] = r["Zab"]; z_line[1,1] = r["Zbb"]; z_line[1,2] = r["Zbc"]
    z_line[2,0] = r["Zac"]; z_line[2,1] = r["Zbc"]; z_line[2,2] = r["Zcc"]

    y_line[0,0] = r["Baa"]; y_line[0,1] = r["Bab"]; y_line[0,2] = r["Bac"]
    y_line[1,0] = r["Bab"]; y_line[1,1] = r["Bbb"]; y_line[1,2] = r["Bbc"]
    y_line[2,0] = r["Bac"]; y_line[2,1] = r["Bbc"]; y_line[2,2] = r["Bcc"]

    t = _to_int(r["type"])
    # Base matrices a,b,c,d
    if t in (1,3):  # line or switch
        a = U + 0.5 * (z_line @ y_line)
        b = z_line.copy()
        c = y_line + 0.25 * (y_line @ z_line @ y_line)
        d = U + 0.5 * (y_line @ z_line)
        A = np.linalg.inv(a)
        B = A @ b

    elif t == 2 and has_transformer:
        tr = _first_row(transformers, lambda x: str(x["config"]) == str(r["config"]))
        if tr is None:
            continue
        conn_h = str(tr["conn_high"]).upper()
        conn_l = str(tr["conn_low"]).upper()
        Zt = tr["Zt"]
        if (conn_h == "GRY" and conn_l == "GRY") or (conn_h == "D" and conn_l == "D"):
            nt = _to_float(tr["kv_high"]) / _to_float(tr["kv_low"])
            a = nt * np.eye(3, dtype=np.complex128)
            b = a * Zt
            c = np.zeros((3,3), dtype=np.complex128)
            d = (1.0/nt) * np.eye(3, dtype=np.complex128)
            A = d
            B = np.diag([Zt, Zt, Zt]).astype(np.complex128)

        elif (conn_h == "D" and conn_l == "GRY"):
            nt = math.sqrt(3.0) * _to_float(tr["kv_high"]) / _to_float(tr["kv_low"])
            a = (-nt/3.0) * np.array([[0,2,1],[1,0,2],[2,1,0]], dtype=np.complex128)
            b = a * Zt
            c = np.zeros((3,3), dtype=np.complex128)
            d = (1.0/nt) * np.array([[ 1,-1, 0],
                                     [ 0, 1,-1],
                                     [-1, 0, 1]], dtype=np.complex128)
            A = (1.0/nt) * np.array([[ 1, 0,-1],
                                     [-1, 1, 0],
                                     [ 0,-1, 1]], dtype=np.complex128)
            B = np.diag([Zt, Zt, Zt]).astype(np.complex128)

        elif (conn_h == "Y" and conn_l == "D"):
            nt = (_to_float(tr["kv_high"]) / math.sqrt(3.0)) / _to_float(tr["kv_low"])
            M = np.array([[ 1,-1, 0],
                          [ 1, 2, 0],
                          [-2,-1, 0]], dtype=np.complex128)
            a = nt * np.array([[ 1,-1, 0],
                               [ 0, 1,-1],
                               [-1, 0, 1]], dtype=np.complex128)
            b = (nt) * (M @ np.eye(3, dtype=np.complex128)) * Zt
            c = np.zeros((3,3), dtype=np.complex128)
            d = (1.0/(3.0*nt)) * M
            A = (1.0/(3.0*nt)) * np.array([[2,1,0],[0,2,1],[1,0,2]], dtype=np.complex128)
            B = np.array([[ Zt,  0,  0],
                          [ 0,  Zt,  0],
                          [-Zt,-Zt,  0]], dtype=np.complex128)
        else:
            err_msg = ("revise transformers.csv file, currently this package only works with "
                       "GrY-GrY, Y-D, D-GrY and D-D three-phase step-down transformer configurations.")
            print(err_msg)
            raise RuntimeError(err_msg)

    elif t == 4 and has_regulator:
        rg = _first_row(regulators, lambda x: str(x["config"]) == str(r["config"]))
        if rg and str(rg["mode"]).upper() == "MANUAL":
            tap1 = _to_float(rg["tap_1"]); tap2 = _to_float(rg["tap_2"]); tap3 = _to_float(rg["tap_3"])
            a = np.diag([1.0/(1.0+0.00625*tap1),
                         1.0/(1.0+0.00625*tap2),
                         1.0/(1.0+0.00625*tap3)]).astype(np.complex128)
            b = np.zeros((3,3), dtype=np.complex128)
            c = np.zeros((3,3), dtype=np.complex128)
            d = np.diag([(1.0+0.00625*tap1),
                         (1.0+0.00625*tap2),
                         (1.0+0.00625*tap3)]).astype(np.complex128)
            A = d
            B = np.zeros((3,3), dtype=np.complex128)
        else:
            a=b=c=d=A=B = np.zeros((3,3), dtype=np.complex128)  # safeguard

    # Store row into gen_lines_mat
    rec = {
        "bus1": _to_int(r["bus1"]),
        "bus2": _to_int(r["bus2"]),
    }
    # Save a,b,c,d and A,B (each 3x3 as individual elements)
    for name, M in (("a",a),("b",b),("c",c),("d",d),("A",A),("B",B)):
        for i in range(3):
            for j in range(3):
                rec[f"{name}_{i+1}_{j+1}"] = np.complex128(M[i,j])
    gen_lines_mat.append(rec)

# ---------- Build loads table ----------
loads = []  # each entry: {"bus":int, "conn":str, "type":str, "ph_1":complex, "ph_2":complex, "ph_3":complex}

# spot loads
wb_id_set = set(b["id"] for b in working_buses)
for r in spot_loads:
    b = _to_int(r["bus"])
    if b in wb_id_set:
        s1 = (_to_float(r["kw_ph1"])  + 1j*_to_float(r["kvar_ph1"])) * 1000.0
        s2 = (_to_float(r["kw_ph2"])  + 1j*_to_float(r["kvar_ph2"])) * 1000.0
        s3 = (_to_float(r["kw_ph3"])  + 1j*_to_float(r["kvar_ph3"])) * 1000.0
        loads.append({"bus": b, "conn": str(r["conn"]).upper(), "type": str(r["type"]).upper(),
                      "ph_1": np.complex128(s1), "ph_2": np.complex128(s2), "ph_3": np.complex128(s3)})

# distributed loads (attach to auxiliary bus busx)
if has_distributed_load and distributed_loads:
    # Prepare mapping bus1,bus2 → busx
    _busx_map = {}
    for a in globals().get("auxiliar_buses", []):
        _busx_map[(int(a["bus1"]), int(a["bus2"]))] = int(a["busx"])
        _busx_map[(int(a["bus2"]), int(a["bus1"]))] = int(a["busx"])  # allow bidirectional keys

    for r in distributed_loads:
        b1 = _to_int(r["bus1"]); b2 = _to_int(r["bus2"])
        target_bus = _busx_map.get((b1, b2), None)

        # Fallback: attach to start bus if mapping missing
        if target_bus is None:
            seg = _first_row(working_segments, lambda s: _to_int(s["bus1"]) == b1 and _to_int(s["bus2"]) == b2)
            if seg is None:
                seg = _first_row(working_segments, lambda s: _to_int(s["bus2"]) == b2)
            target_bus = _to_int(seg["bus1"]) if seg is not None else b1

        s1 = (_to_float(r["kw_ph1"]) + 1j*_to_float(r["kvar_ph1"])) * 1000.0
        s2 = (_to_float(r["kw_ph2"]) + 1j*_to_float(r["kvar_ph2"])) * 1000.0
        s3 = (_to_float(r["kw_ph3"]) + 1j*_to_float(r["kvar_ph3"])) * 1000.0

        loads.append({"bus": target_bus, "conn": str(r["conn"]).upper(), "type": str(r["type"]).upper(),
                      "ph_1": np.complex128(s1), "ph_2": np.complex128(s2), "ph_3": np.complex128(s3)})

# capacitors → negative reactive power (Y, Z)
for r in input_capacitors:
    b = _to_int(r["bus"])
    if b in wb_id_set:
        s1 = -1j * _to_float(r["kvar_ph1"]) * 1000.0
        s2 = -1j * _to_float(r["kvar_ph2"]) * 1000.0
        s3 = -1j * _to_float(r["kvar_ph3"]) * 1000.0
        loads.append({"bus": b, "conn": "Y", "type": "Z",
                      "ph_1": np.complex128(s1), "ph_2": np.complex128(s2), "ph_3": np.complex128(s3)})

# ---------- Identify/load distributed generation (DG) ----------
has_pq_distributed_gen  = False
has_pqv_distributed_gen = False
has_pi_distributed_gen  = False
pq_distributed_gen  = []
pqv_distributed_gen = []
pi_distributed_gen  = []
generation_register = []  # {"bus","mode","conn","kw_ph1","kvar_ph1",...,"kW/kVAr per phase", "max_diff"}

if has_distributed_gen and distributed_gen:
    # PQ
    pq = [r for r in distributed_gen if str(r["mode"]).upper() == "PQ"]
    # Drop rows with missing data
    _valid = []
    for r in pq:
        if not (_is_missing(r.get("kw_set")) or _is_missing(r.get("kvar_set"))):
            _valid.append(r)
        else:
            print("PQ distributed generation with missing values, it will be ignored.")
    pq = [r for r in _valid if _to_int(r.get("bus")) in wb_id_set]  # only existing buses
    if len(pq) > 0:
        for r in pq:
            b = _to_int(r["bus"]); conn = str(r["conn"]).upper()
            s_phase = (_to_float(r["kw_set"]) + 1j*_to_float(r["kvar_set"])) * 1000.0 / 3.0
            loads.append({"bus": b, "conn": conn, "type": "PQ",
                          "ph_1": -np.complex128(s_phase),
                          "ph_2": -np.complex128(s_phase),
                          "ph_3": -np.complex128(s_phase)})
            generation_register.append({"bus": b, "mode": "PQ", "conn": conn,
                                        "kw_ph1": _to_float(r["kw_set"])/3.0, "kvar_ph1": _to_float(r["kvar_set"])/3.0,
                                        "kw_ph2": _to_float(r["kw_set"])/3.0, "kvar_ph2": _to_float(r["kvar_set"])/3.0,
                                        "kw_ph3": _to_float(r["kw_set"])/3.0, "kvar_ph3": _to_float(r["kvar_set"])/3.0,
                                        "max_diff": 0.0})
        has_pq_distributed_gen = True

    # PQV
    pqv = [r for r in distributed_gen if str(r["mode"]).upper() == "PQV"]
    _valid = []
    for r in pqv:
        if any(_is_missing(r.get(k)) for k in ("kw_set","kv_set","kvar_min","kvar_max","xd")):
            print("PQV distributed generation register with missing values, it will be ignored.")
        else:
            _valid.append(r)
    pqv = [r for r in _valid if _to_int(r.get("bus")) in wb_id_set]
    if len(pqv) > 0:
        for r in pqv:
            b = _to_int(r["bus"]); conn = str(r["conn"]).upper()
            p_phase = _to_float(r["kw_set"]) * 1000.0 / 3.0
            q_phase = (_to_float(r["kvar_min"]) + _to_float(r["kvar_max"])) * 1000.0 / 6.0
            s_phase = p_phase + 1j*q_phase
            loads.append({"bus": b, "conn": conn, "type": "PQV",
                          "ph_1": -np.complex128(s_phase),
                          "ph_2": -np.complex128(s_phase),
                          "ph_3": -np.complex128(s_phase)})
            generation_register.append({"bus": b, "mode": "PQV", "conn": conn,
                                        "kw_ph1": _to_float(r["kw_set"])/3.0, "kvar_ph1": (_to_float(r["kvar_min"])+_to_float(r["kvar_max"])) / 3.0,
                                        "kw_ph2": _to_float(r["kw_set"])/3.0, "kvar_ph2": (_to_float(r["kvar_min"])+_to_float(r["kvar_max"])) / 3.0,
                                        "kw_ph3": _to_float(r["kw_set"])/3.0, "kvar_ph3": (_to_float(r["kvar_min"])+_to_float(r["kvar_max"])) / 3.0,
                                        "max_diff": 0.0})
        # State tracking columns initialised to 0
        for r in pqv:
            pqv_distributed_gen.append({"bus": _to_int(r["bus"]), "conn": str(r["conn"]).upper(), "mode": "PQV",
                                        "kw_set": _to_float(r["kw_set"]), "kv_set": _to_float(r["kv_set"]),
                                        "kvar_min": _to_float(r["kvar_min"]), "kvar_max": _to_float(r["kvar_max"]),
                                        "xd": _to_float(r["xd"]),
                                        "v_ph1": 0.0, "v_ph2": 0.0, "v_ph3": 0.0, "max_diff": 0.0,
                                        "w_ph1": 0.0, "w_ph2": 0.0, "w_ph3": 0.0,
                                        "var_ph1": 0.0, "var_ph2": 0.0, "var_ph3": 0.0})
        has_pqv_distributed_gen = True

    # PI
    pi = [r for r in distributed_gen if str(r["mode"]).upper() == "PI"]
    _valid = []
    for r in pi:
        if any(_is_missing(r.get(k)) for k in ("kw_set","amp_set","kvar_min","kvar_max")):
            print("PI distributed generation register with missing values, it will be ignored.")
        else:
            _valid.append(r)
    pi = [r for r in _valid if _to_int(r.get("bus")) in wb_id_set]
    if len(pi) > 0:
        for r in pi:
            b = _to_int(r["bus"]); conn = str(r["conn"]).upper()
            p_phase = _to_float(r["kw_set"]) * 1000.0 / 3.0
            q_phase = (_to_float(r["kvar_min"]) + _to_float(r["kvar_max"])) * 1000.0 / 6.0
            s_phase = p_phase + 1j*q_phase
            loads.append({"bus": b, "conn": conn, "type": "PI",
                          "ph_1": -np.complex128(s_phase),
                          "ph_2": -np.complex128(s_phase),
                          "ph_3": -np.complex128(s_phase)})
            generation_register.append({"bus": b, "mode": "PI", "conn": conn,
                                        "kw_ph1": _to_float(r["kw_set"])/3.0, "kvar_ph1": (_to_float(r["kvar_min"])+_to_float(r["kvar_max"])) / 3.0,
                                        "kw_ph2": _to_float(r["kw_set"])/3.0, "kvar_ph2": (_to_float(r["kvar_min"])+_to_float(r["kvar_max"])) / 3.0,
                                        "kw_ph3": _to_float(r["kw_set"])/3.0, "kvar_ph3": (_to_float(r["kvar_min"])+_to_float(r["kvar_max"])) / 3.0,
                                        "max_diff": 0.0})
        for r in pi:
            pi_distributed_gen.append({"bus": _to_int(r["bus"]), "conn": str(r["conn"]).upper(), "mode": "PI",
                                       "kw_set": _to_float(r["kw_set"]), "amp_set": _to_float(r["amp_set"]),
                                       "kvar_min": _to_float(r["kvar_min"]), "kvar_max": _to_float(r["kvar_max"]),
                                       "v_ph1": 0.0, "v_ph2": 0.0, "v_ph3": 0.0, "max_diff": 0.0,
                                       "w_ph1": 0.0, "w_ph2": 0.0, "w_ph3": 0.0,
                                       "var_ph1": 0.0, "var_ph2": 0.0, "var_ph3": 0.0})
        has_pi_distributed_gen = True

    # If no DG types exist, set overall flag to false
    if not (has_pq_distributed_gen or has_pqv_distributed_gen or has_pi_distributed_gen):
        has_distributed_gen = False

# Add constants k_1~k_3 to loads (Z:I conversion)
for ld in loads:
    ld["k_1"] = None; ld["k_2"] = None; ld["k_3"] = None
    if ld["type"] in ("Z","I"):
        # Determine bus base voltage (Y: V_LN, Δ: √3·V_LN)
        vb = None
        wb = _first_row(working_buses, lambda b: b["id"] == ld["bus"])
        if wb:
            vb = float(wb["v_base"])
            if ld["conn"] == "D":
                vb *= math.sqrt(3.0)
        if ld["type"] == "Z":
            # k = V_nom^2 / |S| · exp(j·∠S)
            for kname, sph in (("k_1","ph_1"),("k_2","ph_2"),("k_3","ph_3")):
                S = ld[sph]
                if S == 0 or S == 0.0:
                    ld[kname] = 0.0+0.0j
                else:
                    mag = abs(S)
                    ang = np.angle(S)
                    ld[kname] = np.complex128((vb**2 / mag) * np.exp(1j*ang))
        elif ld["type"] == "I":
            # k = |S| / V_nom  (current magnitude [A])
            for kname, sph in (("k_1","ph_1"),("k_2","ph_2"),("k_3","ph_3")):
                S = ld[sph]
                if S == 0 or S == 0.0:
                    ld[kname] = 0.0
                else:
                    ld[kname] = np.float64(abs(S) / vb)

# Add process(Int8), phases, v_ph1~3, ibus_1~3 to working_buses
for b in working_buses:
    b["process"] = np.int8(0)
    b["phases"]  = None
    b["v_ph1"] = np.complex128(0+0j)
    b["v_ph2"] = np.complex128(0+0j)
    b["v_ph3"] = np.complex128(0+0j)
# Sort by number descending (original rev=true)
working_buses.sort(key=lambda x: x["number"], reverse=True)
for b in working_buses:
    b["ibus_1"] = np.complex128(0+0j)
    b["ibus_2"] = np.complex128(0+0j)
    b["ibus_3"] = np.complex128(0+0j)

# Initialise lines ibus1_1~3
for r in lines:
    r["ibus1_1"] = np.complex128(0+0j)
    r["ibus1_2"] = np.complex128(0+0j)
    r["ibus1_3"] = np.complex128(0+0j)

# Temporary containers for bus voltages/currents (three-phase)
Vbus1 = np.zeros(3, dtype=np.complex128)
Vbus2 = np.zeros(3, dtype=np.complex128)
Ibus1 = np.zeros(3, dtype=np.complex128)
Ibus2 = np.zeros(3, dtype=np.complex128)



# === power_flow ===
# Purpose: implement outer iteration (PQV/PI DG correction) and final CSV generation.
# Prerequisite: earlier blocks (data_input → topology_discovery →
# data_preparation → sweep_procedures) have already prepared the global state.

# ---------- Execution parameters (defaults if undefined above) ----------
try:
    tolerance
except NameError:
    tolerance = 1e-6
try:
    max_iterations
except NameError:
    max_iterations = 30
try:
    display_summary
except NameError:
    display_summary = True
try:
    timestamp
except NameError:
    timestamp = False

# ---------- Helpers ----------
def _deg(z):  # complex angle in degrees
    return np.rad2deg(np.angle(z))
def _safe_rel_diff(old: np.ndarray, new: np.ndarray):
    old = np.asarray(old, dtype=np.float64)
    new = np.asarray(new, dtype=np.float64)
    denom = np.where(np.abs(new) > 0, np.abs(new), 1.0)
    return np.max(np.abs((old - new) / denom))
def _dump_complex_as_mag_deg(writer, zs):  # record (magnitude, degree) pairs
    for z in zs:
        writer.writerow([np.abs(z), _deg(z)])

# =====================================================================
# External iteration loop (adjust PQV/PI DG; PQ fixed in data_preparation)
# =====================================================================
outer_iteration = 0
max_diff = 1.0
inner_iteration = 0  # inner iteration count of last outer round

while max_diff > tolerance:
    # ---------------------------
    # 1) Run FBS internal loop until convergence
    #    (reuse while loop from sweep_procedures block)
    # ---------------------------
    # Initialization
    max_error = 1.0
    iter_number = 0

    while max_error > tolerance:
        iter_number += 1

        # ---- forward sweep ----
        for b in working_buses:
            if b["type"] == 1:
                b["v_ph1"] = np.complex128(ELN[0])
                b["v_ph2"] = np.complex128(ELN[1])
                b["v_ph3"] = np.complex128(ELN[2])
                b["process"] = np.int8(1)

        for n in range(len(working_buses)-1, -1, -1):
            b1_id = int(working_buses[n]["id"])
            for m in range(len(gen_lines_mat)):
                gl = gen_lines_mat[m]
                if int(gl["bus1"]) != b1_id:
                    continue
                b2_id = int(gl["bus2"])
                n2 = next((i for i, wb in enumerate(working_buses) if int(wb["id"]) == b2_id), None)
                if n2 is None:
                    continue

                A = np.array([[gl["A_1_1"], gl["A_1_2"], gl["A_1_3"]],
                              [gl["A_2_1"], gl["A_2_2"], gl["A_2_3"]],
                              [gl["A_3_1"], gl["A_3_2"], gl["A_3_3"]]], dtype=np.complex128)
                B = np.array([[gl["B_1_1"], gl["B_1_2"], gl["B_1_3"]],
                              [gl["B_2_1"], gl["B_2_2"], gl["B_2_3"]],
                              [gl["B_3_1"], gl["B_3_2"], gl["B_3_3"]]], dtype=np.complex128)

                Vbus1[0] = working_buses[n]["v_ph1"]
                Vbus1[1] = working_buses[n]["v_ph2"]
                Vbus1[2] = working_buses[n]["v_ph3"]
                Ibus2[0] = working_buses[n2]["ibus_1"]
                Ibus2[1] = working_buses[n2]["ibus_2"]
                Ibus2[2] = working_buses[n2]["ibus_3"]

                Vbus2[:] = A @ Vbus1 - B @ Ibus2
                working_buses[n2]["v_ph1"] = np.complex128(Vbus2[0])
                working_buses[n2]["v_ph2"] = np.complex128(Vbus2[1])
                working_buses[n2]["v_ph3"] = np.complex128(Vbus2[2])
                working_buses[n2]["process"] = np.int8(1)

        # ---- backward sweep ----
        a = np.zeros((3,3), dtype=np.complex128)
        b = np.zeros((3,3), dtype=np.complex128)
        c = np.zeros((3,3), dtype=np.complex128)
        d = np.zeros((3,3), dtype=np.complex128)
        Iline  = np.zeros(3, dtype=np.complex128)
        Iphase = np.zeros(3, dtype=np.complex128)
        DL = np.array([[ 1,  0, -1],
                       [-1,  1,  0],
                       [ 0, -1,  1]], dtype=np.float64)

        for b_ in working_buses:
            b_["ibus_1"] = np.complex128(0+0j)
            b_["ibus_2"] = np.complex128(0+0j)
            b_["ibus_3"] = np.complex128(0+0j)

        # Terminal buses: load currents
        for n in range(len(working_buses)):
            bn = working_buses[n]
            if bn["type"] != 5:
                continue
            bus_id = int(bn["id"])
            for ld in loads:
                if int(ld["bus"]) != bus_id:
                    continue
                if ld["conn"] == "Y":
                    if ld["type"] in ("PQ","PQV","PI"):
                        bn["ibus_1"] += np.conj(ld["ph_1"] / bn["v_ph1"])
                        bn["ibus_2"] += np.conj(ld["ph_2"] / bn["v_ph2"])
                        bn["ibus_3"] += np.conj(ld["ph_3"] / bn["v_ph3"])
                    if ld["type"] == "Z":
                        if ld["k_1"] != 0: bn["ibus_1"] += bn["v_ph1"] / ld["k_1"]
                        if ld["k_2"] != 0: bn["ibus_2"] += bn["v_ph2"] / ld["k_2"]
                        if ld["k_3"] != 0: bn["ibus_3"] += bn["v_ph3"] / ld["k_3"]
                    if ld["type"] == "I":
                        bn["ibus_1"] += np.abs(ld["k_1"]) * np.exp(1j*(np.angle(bn["v_ph1"]) - np.angle(ld["ph_1"])))
                        bn["ibus_2"] += np.abs(ld["k_2"]) * np.exp(1j*(np.angle(bn["v_ph2"]) - np.angle(ld["ph_2"])))
                        bn["ibus_3"] += np.abs(ld["k_3"]) * np.exp(1j*(np.angle(bn["v_ph3"]) - np.angle(ld["ph_3"])))
                if ld["conn"] == "D":
                    if ld["type"] in ("PQ","PQV","PI"):
                        Iphase[0] = np.conj(ld["ph_1"] / (bn["v_ph1"] - bn["v_ph2"]))
                        Iphase[1] = np.conj(ld["ph_2"] / (bn["v_ph2"] - bn["v_ph3"]))
                        Iphase[2] = np.conj(ld["ph_3"] / (bn["v_ph3"] - bn["v_ph1"]))
                    if ld["type"] == "Z":
                        Iphase[:] = 0
                        if ld["k_1"] != 0: Iphase[0] = (bn["v_ph1"] - bn["v_ph2"]) / ld["k_1"]
                        if ld["k_2"] != 0: Iphase[1] = (bn["v_ph2"] - bn["v_ph3"]) / ld["k_2"]
                        if ld["k_3"] != 0: Iphase[2] = (bn["v_ph3"] - bn["v_ph1"]) / ld["k_3"]
                    if ld["type"] == "I":
                        Iphase[0] = np.abs(ld["k_1"]) * np.exp(1j*(np.angle(bn["v_ph1"] - bn["v_ph2"]) - np.angle(ld["ph_1"])))
                        Iphase[1] = np.abs(ld["k_2"]) * np.exp(1j*(np.angle(bn["v_ph2"] - bn["v_ph3"]) - np.angle(ld["ph_2"])))
                        Iphase[2] = np.abs(ld["k_3"]) * np.exp(1j*(np.angle(bn["v_ph3"] - bn["v_ph1"]) - np.angle(ld["ph_3"])))
                    Iline[:] = DL @ Iphase
                    bn["ibus_1"] += Iline[0]
                    bn["ibus_2"] += Iline[1]
                    bn["ibus_3"] += Iline[2]
                    Iline[:] = 0+0j
                    Iphase[:] = 0+0j
            bn["process"] = np.int8(2)

        # Non-terminal: convert child → parent and add local load
        for n in range(len(working_buses)):
            bn = working_buses[n]
            if bn["type"] == 5:
                continue
            bus_id = int(bn["id"])
            for m in range(len(gen_lines_mat)):
                gl = gen_lines_mat[m]
                if int(gl["bus1"]) != bus_id:
                    continue
                b2_id = int(gl["bus2"])
                n2 = next((i for i, wb in enumerate(working_buses) if int(wb["id"]) == b2_id), None)
                if n2 is None:
                    continue
                a = np.array([[gl["a_1_1"], gl["a_1_2"], gl["a_1_3"]],
                              [gl["a_2_1"], gl["a_2_2"], gl["a_2_3"]],
                              [gl["a_3_1"], gl["a_3_2"], gl["a_3_3"]]], dtype=np.complex128)
                b = np.array([[gl["b_1_1"], gl["b_1_2"], gl["b_1_3"]],
                              [gl["b_2_1"], gl["b_2_2"], gl["b_2_3"]],
                              [gl["b_3_1"], gl["b_3_2"], gl["b_3_3"]]], dtype=np.complex128)
                c = np.array([[gl["c_1_1"], gl["c_1_2"], gl["c_1_3"]],
                              [gl["c_2_1"], gl["c_2_2"], gl["c_2_3"]],
                              [gl["c_3_1"], gl["c_3_2"], gl["c_3_3"]]], dtype=np.complex128)
                d = np.array([[gl["d_1_1"], gl["d_1_2"], gl["d_1_3"]],
                              [gl["d_2_1"], gl["d_2_2"], gl["d_2_3"]],
                              [gl["d_3_1"], gl["d_3_2"], gl["d_3_3"]]], dtype=np.complex128)

                Vbus2[0] = working_buses[n2]["v_ph1"]
                Vbus2[1] = working_buses[n2]["v_ph2"]
                Vbus2[2] = working_buses[n2]["v_ph3"]
                Ibus2[0] = working_buses[n2]["ibus_1"]
                Ibus2[1] = working_buses[n2]["ibus_2"]
                Ibus2[2] = working_buses[n2]["ibus_3"]

                Vbus1[:] = a @ Vbus2 + b @ Ibus2
                Ibus1[:] = c @ Vbus2 + d @ Ibus2

                bn["v_ph1"] = np.complex128(Vbus1[0])
                bn["v_ph2"] = np.complex128(Vbus1[1])
                bn["v_ph3"] = np.complex128(Vbus1[2])
                bn["ibus_1"] += Ibus1[0]
                bn["ibus_2"] += Ibus1[1]
                bn["ibus_3"] += Ibus1[2]
                lines[m]["ibus1_1"] = np.complex128(Ibus1[0])
                lines[m]["ibus1_2"] = np.complex128(Ibus1[1])
                lines[m]["ibus1_3"] = np.complex128(Ibus1[2])
                bn["process"] = np.int8(2)

            # Add local load
            for ld in loads:
                if int(ld["bus"]) != bus_id:
                    continue
                if ld["conn"] == "Y":
                    if ld["type"] in ("PQ","PQV","PI"):
                        bn["ibus_1"] += np.conj(ld["ph_1"] / bn["v_ph1"])
                        bn["ibus_2"] += np.conj(ld["ph_2"] / bn["v_ph2"])
                        bn["ibus_3"] += np.conj(ld["ph_3"] / bn["v_ph3"])
                    if ld["type"] == "Z":
                        if ld["k_1"] != 0: bn["ibus_1"] += bn["v_ph1"] / ld["k_1"]
                        if ld["k_2"] != 0: bn["ibus_2"] += bn["v_ph2"] / ld["k_2"]
                        if ld["k_3"] != 0: bn["ibus_3"] += bn["v_ph3"] / ld["k_3"]
                    if ld["type"] == "I":
                        bn["ibus_1"] += np.abs(ld["k_1"]) * np.exp(1j*(np.angle(bn["v_ph1"]) - np.angle(ld["ph_1"])))
                        bn["ibus_2"] += np.abs(ld["k_2"]) * np.exp(1j*(np.angle(bn["v_ph2"]) - np.angle(ld["ph_2"])))
                        bn["ibus_3"] += np.abs(ld["k_3"]) * np.exp(1j*(np.angle(bn["v_ph3"]) - np.angle(ld["ph_3"])))
                if ld["conn"] == "D":
                    if ld["type"] in ("PQ","PQV","PI"):
                        Iphase[0] = np.conj(ld["ph_1"] / (bn["v_ph1"] - bn["v_ph2"]))
                        Iphase[1] = np.conj(ld["ph_2"] / (bn["v_ph2"] - bn["v_ph3"]))
                        Iphase[2] = np.conj(ld["ph_3"] / (bn["v_ph3"] - bn["v_ph1"]))
                    if ld["type"] == "Z":
                        Iphase[:] = 0
                        if ld["k_1"] != 0: Iphase[0] = (bn["v_ph1"] - bn["v_ph2"]) / ld["k_1"]
                        if ld["k_2"] != 0: Iphase[1] = (bn["v_ph2"] - bn["v_ph3"]) / ld["k_2"]
                        if ld["k_3"] != 0: Iphase[2] = (bn["v_ph3"] - bn["v_ph1"]) / ld["k_3"]
                    if ld["type"] == "I":
                        Iphase[0] = np.abs(ld["k_1"]) * np.exp(1j*(np.angle(bn["v_ph1"] - bn["v_ph2"]) - np.angle(ld["ph_1"])))
                        Iphase[1] = np.abs(ld["k_2"]) * np.exp(1j*(np.angle(bn["v_ph2"] - bn["v_ph3"]) - np.angle(ld["ph_2"])))
                        Iphase[2] = np.abs(ld["k_3"]) * np.exp(1j*(np.angle(bn["v_ph3"] - bn["v_ph1"]) - np.angle(ld["ph_3"])))
                    Iline[:] = DL @ Iphase
                    bn["ibus_1"] += Iline[0]
                    bn["ibus_2"] += Iline[1]
                    bn["ibus_3"] += Iline[2]
                    Iline[:] = 0+0j
                    Iphase[:] = 0+0j

        # 3) Evaluate substation voltage error
        _sub_id = int(substation[0]["bus"])
        _sub = next(b for b in working_buses if int(b["id"]) == _sub_id)
        sub_phase = np.array([_sub["v_ph1"], _sub["v_ph2"], _sub["v_ph3"]], dtype=np.complex128)
        sub_line = D @ sub_phase
        errs = np.abs((np.abs(ELL) - np.abs(sub_line)) / np.abs(ELL))
        max_error = float(np.max(errs))

        if iter_number == max_iterations:
            print(f"Program halted: maximum number of forward-backward iterations reached ({max_iterations})")
            break

    # Record inner iterations for this outer round
    inner_iteration = iter_number

    # ---------------------------
    # 2) DG adjustment: PQV, PI
    # ---------------------------
    if not has_distributed_gen:
        max_diff = 0.0
        outer_iteration += 1
        break

    # (1) PQV
    if has_pqv_distributed_gen and len(pqv_distributed_gen) > 0:
        # Gather current voltage magnitude
        id2V = {int(wb["id"]): (abs(wb["v_ph1"]), abs(wb["v_ph2"]), abs(wb["v_ph3"])) for wb in working_buses}

        for rec in pqv_distributed_gen:
            bus = int(rec["bus"])
            v_old = np.array([rec["v_ph1"], rec["v_ph2"], rec["v_ph3"]], dtype=np.float64)
            v_new = np.array(id2V[bus], dtype=np.float64)

            v_set = float(rec["kv_set"]) * 1000.0 / math.sqrt(3.0)
            xd    = float(rec["xd"])
            p_ph  = float(rec["kw_set"]) * 1000.0 / 3.0

            # Formula: var = sqrt((v_set*|v|/xd)^2 - p_ph^2) - |v|^2/xd (per phase)
            # Numerical safety: treat negative term under sqrt as 0
            var = []
            for k in range(3):
                term = (v_set * v_new[k] / xd)**2 - p_ph**2
                term = term if term > 0 else 0.0
                var_k = math.sqrt(term) - (v_new[k]**2) / xd
                var.append(var_k)
            var = np.asarray(var, dtype=np.float64)

            # Clamp to limits
            qmin = float(rec["kvar_min"]) * 1000.0 / 3.0
            qmax = float(rec["kvar_max"]) * 1000.0 / 3.0
            var = np.clip(var, qmin, qmax)

            # Remove existing DG row from loads (type == "PQV" & bus == rec["bus"])
            loads = [ld for ld in loads if not (int(ld["bus"]) == bus and ld["type"] == "PQV")]
            # Insert new DG row (per-phase S = P + jQ, injection negative)
            s_ph = p_ph + 1j*var
            loads.append({"bus": bus, "conn": rec["conn"], "type": "PQV",
                          "ph_1": -np.complex128(s_ph[0]),
                          "ph_2": -np.complex128(s_ph[1]),
                          "ph_3": -np.complex128(s_ph[2])})

            # Record change ratio
            max_volt_diff = _safe_rel_diff(v_old, v_new)
            rec["v_ph1"], rec["v_ph2"], rec["v_ph3"] = float(v_new[0]), float(v_new[1]), float(v_new[2])
            rec["max_diff"] = float(max_volt_diff)
            rec["w_ph1"] = p_ph; rec["w_ph2"] = p_ph; rec["w_ph3"] = p_ph
            rec["var_ph1"] = float(var[0]); rec["var_ph2"] = float(var[1]); rec["var_ph3"] = float(var[2])

            # Update generation_register (one row per bus)
            generation_register = [g for g in generation_register if int(g["bus"]) != bus]
            generation_register.append({"bus": bus, "mode": "PQV", "conn": rec["conn"],
                                        "kw_ph1": p_ph/1000.0, "kvar_ph1": var[0]/1000.0,
                                        "kw_ph2": p_ph/1000.0, "kvar_ph2": var[1]/1000.0,
                                        "kw_ph3": p_ph/1000.0, "kvar_ph3": var[2]/1000.0,
                                        "max_diff": float(max_volt_diff)})

    # (2) PI
    if has_pi_distributed_gen and len(pi_distributed_gen) > 0:
        id2V = {int(wb["id"]): (wb["v_ph1"], wb["v_ph2"], wb["v_ph3"]) for wb in working_buses}

        for rec in pi_distributed_gen:
            bus   = int(rec["bus"])
            v_bus = id2V[bus]
            v_new = np.array([abs(v_bus[0]), abs(v_bus[1]), abs(v_bus[2])], dtype=np.float64)
            v_old = np.array([rec["v_ph1"], rec["v_ph2"], rec["v_ph3"]], dtype=np.float64)

            i_set = float(rec["amp_set"])
            p_ph  = float(rec["kw_set"]) * 1000.0 / 3.0
            qmin  = float(rec["kvar_min"]) * 1000.0 / 3.0
            qmax  = float(rec["kvar_max"]) * 1000.0 / 3.0

            # Reference voltage/current per connection
            if rec["conn"] == "Y":
                v_ph = v_new.copy()
                i_ph = i_set
            else:  # "D"
                # Line-to-line voltage magnitude
                v_ph = np.array([abs(v_bus[0]-v_bus[1]),
                                 abs(v_bus[1]-v_bus[2]),
                                 abs(v_bus[2]-v_bus[0])], dtype=np.float64)
                i_ph = i_set / math.sqrt(3.0)

            # q_ph = sqrt((i_ph*v)^2 - p_ph^2); negative → qmin
            q_ph = []
            for k in range(3):
                term = (i_ph * v_ph[k])**2 - p_ph**2
                qk = math.sqrt(term) if term >= 0 else qmin
                q_ph.append(qk)
            q_ph = np.clip(np.array(q_ph, dtype=np.float64), qmin, qmax)

            # Replace loads (type=="PI")
            loads = [ld for ld in loads if not (int(ld["bus"]) == bus and ld["type"] == "PI")]
            s_ph = p_ph + 1j*q_ph
            loads.append({"bus": bus, "conn": rec["conn"], "type": "PI",
                          "ph_1": -np.complex128(s_ph[0]),
                          "ph_2": -np.complex128(s_ph[1]),
                          "ph_3": -np.complex128(s_ph[2])})

            max_volt_diff = _safe_rel_diff(v_old, v_new)
            rec["v_ph1"], rec["v_ph2"], rec["v_ph3"] = float(v_new[0]), float(v_new[1]), float(v_new[2])
            rec["max_diff"] = float(max_volt_diff)
            rec["w_ph1"] = p_ph; rec["w_ph2"] = p_ph; rec["w_ph3"] = p_ph
            rec["var_ph1"] = float(q_ph[0]); rec["var_ph2"] = float(q_ph[1]); rec["var_ph3"] = float(q_ph[2])

            generation_register = [g for g in generation_register if int(g["bus"]) != bus]
            generation_register.append({"bus": bus, "mode": "PI", "conn": rec["conn"],
                                        "kw_ph1": p_ph/1000.0, "kvar_ph1": q_ph[0]/1000.0,
                                        "kw_ph2": p_ph/1000.0, "kvar_ph2": q_ph[1]/1000.0,
                                        "kw_ph3": p_ph/1000.0, "kvar_ph3": q_ph[2]/1000.0,
                                        "max_diff": float(max_volt_diff)})

    # (3) Termination check
    if len(generation_register) > 0:
        max_diff = max(float(g["max_diff"]) for g in generation_register)
    else:
        max_diff = 0.0

    outer_iteration += 1

# === results ===
# Required globals: working_buses, lines, gen_lines_mat, substation, D, ELL, output_dir,
#                   has_distributed_load, auxiliar_buses (if any), has_distributed_gen, generation_register,
#            display_summary(bool), timestamp(bool)
def _deg(z): return float(np.rad2deg(np.angle(z)))
def _round(x, d): return (None if x is None else (round(float(x), d)))
def _out(name):
    ts = "_" + datetime.now().strftime("%Y%m%d-%H%M") if 'timestamp' in globals() and timestamp else ""
    return os.path.join(output_dir, f"{name}{ts}.csv")

# 0) Populate phases for each bus (root uses 'abc')
for wb in working_buses:
    wb["phases"] = None
root_id = int(substation[0]["bus"])
for wb in working_buses:
    if int(wb["id"]) == root_id:
        wb["phases"] = "abc"
for ln in lines:
    bus2 = int(ln["bus2"])
    phs  = ln.get("phases")
    if phs:
        for wb in working_buses:
            if int(wb["id"]) == bus2 and wb["phases"] is None:
                wb["phases"] = phs

# Add phases for auxiliary buses if present
if 'has_distributed_load' in globals() and has_distributed_load and auxiliar_buses:
    for aux in auxiliar_buses:
        bx = int(aux["busx"])
        ph = next((wb["phases"] for wb in working_buses if int(wb["id"]) == bx), None)
        aux["phases"] = ph

# 1) Volts (phase LN) & Volts p.u.
wb_sorted = sorted(working_buses, key=lambda x:int(x["id"]))
volts_phase_rows = []
volts_pu_rows    = []
ext_max_pu = 0.0; ext_max_bus = 0
ext_min_pu = 2.0; ext_min_bus = 0

for wb in wb_sorted:
    vid = int(wb["id"]); phs = wb.get("phases") or "abc"
    vA, vB, vC = wb["v_ph1"], wb["v_ph2"], wb["v_ph3"]
    vb = float(wb["v_base"]) if wb.get("v_base") is not None else np.nan
    # phase volt & pu with selective filling by available phases
    rowP = [vid, None, None, None, None, None, None]
    rowPU = [vid, None, None, None, None, None, None]  # id, volt_A, deg_A, volt_B, deg_B, volt_C, deg_C

    if "a" in phs:
        rowP[1] = _round(abs(vA), 1);
        rowP[2] = _round(_deg(vA), 2)
        rowPU[1] = _round(abs(vA) / vb, 4)
        rowPU[2] = _round(_deg(vA), 2)
        if rowPU[1] is not None:
            if rowPU[1] > ext_max_pu: ext_max_pu, ext_max_bus = rowPU[1], vid
            if rowPU[1] < ext_min_pu: ext_min_pu, ext_min_bus = rowPU[1], vid

    if "b" in phs:
        rowP[3] = _round(abs(vB), 1);
        rowP[4] = _round(_deg(vB), 2)
        rowPU[3] = _round(abs(vB) / vb, 4)
        rowPU[4] = _round(_deg(vB), 2)
        if rowPU[3] is not None:
            if rowPU[3] > ext_max_pu: ext_max_pu, ext_max_bus = rowPU[3], vid
            if rowPU[3] < ext_min_pu: ext_min_pu, ext_min_bus = rowPU[3], vid

    if "c" in phs:
        rowP[5] = _round(abs(vC), 1);
        rowP[6] = _round(_deg(vC), 2)
        rowPU[5] = _round(abs(vC) / vb, 4)
        rowPU[6] = _round(_deg(vC), 2)
        if rowPU[5] is not None:
            if rowPU[5] > ext_max_pu: ext_max_pu, ext_max_bus = rowPU[5], vid
            if rowPU[5] < ext_min_pu: ext_min_pu, ext_min_bus = rowPU[5], vid

    volts_phase_rows.append(rowP)
    volts_pu_rows.append(rowPU)

# 2) Volts line (LL) — optional
volts_line_rows = []
for wb in wb_sorted:
    vid = int(wb["id"]); phs = wb.get("phases") or "abc"
    v1, v2, v3 = wb["v_ph1"], wb["v_ph2"], wb["v_ph3"]
    AB = v1 - v2; BC = v2 - v3; CA = v3 - v1
    row = [vid, None, None, None, None, None, None]
    if "a" in phs and "b" in phs:
        row[1] = _round(abs(AB), 1); row[2] = _round(_deg(AB), 2)
    if "b" in phs and "c" in phs:
        row[3] = _round(abs(BC), 1); row[4] = _round(_deg(BC), 2)
    if "c" in phs and "a" in phs:
        row[5] = _round(abs(CA), 1); row[6] = _round(_deg(CA), 2)
    volts_line_rows.append(row)

# 3) Line current/complex power (in/out) and losses
# in_I_ph* from lines[*]["ibus1_*"]; out_I_ph* = working_buses at 'to' (JL style)
id2wb = {int(w["id"]): w for w in working_buses}
cflow_rows = []   # current flow (magn/deg)
pflow_rows = []   # power flow (kW/kVAr)
loss_rows = []    # losses per phase & totals (kW/kVAr)
for m, ln in enumerate(lines):
    b1 = int(ln["bus1"]); b2 = int(ln["bus2"]); phs = ln.get("phases") or "abc"
    inI = np.array([ln["ibus1_1"], ln["ibus1_2"], ln["ibus1_3"]], dtype=np.complex128)
    outI= np.array([id2wb[b2]["ibus_1"], id2wb[b2]["ibus_2"], id2wb[b2]["ibus_3"]], dtype=np.complex128)
    V1  = np.array([id2wb[b1]["v_ph1"], id2wb[b1]["v_ph2"], id2wb[b1]["v_ph3"]], dtype=np.complex128)
    V2  = np.array([id2wb[b2]["v_ph1"], id2wb[b2]["v_ph2"], id2wb[b2]["v_ph3"]], dtype=np.complex128)
    inS = V1 * np.conj(inI); outS = V2 * np.conj(outI)
    # current report (optional)
    crow = [b1, b2] + [None]*12
    if "a" in phs:
        crow[2] = _round(abs(inI[0]), 2); crow[3] = _round(_deg(inI[0]), 2)
        crow[8] = _round(abs(outI[0]), 2); crow[9] = _round(_deg(outI[0]), 2)
    if "b" in phs:
        crow[4] = _round(abs(inI[1]), 2); crow[5] = _round(_deg(inI[1]), 2)
        crow[10]= _round(abs(outI[1]), 2); crow[11]= _round(_deg(outI[1]), 2)
    if "c" in phs:
        crow[6] = _round(abs(inI[2]), 2); crow[7] = _round(_deg(inI[2]), 2)
        crow[12]= _round(abs(outI[2]), 2); crow[13]= _round(_deg(outI[2]), 2)
    cflow_rows.append(crow)
    # power report (kW/kVAr, optional)
    prow = [b1, b2] + [None]*12
    def _kw(z): return _round(np.real(z)/1000.0, 3)
    def _kvar(z): return _round(np.imag(z)/1000.0, 3)
    if "a" in phs:
        prow[2],  prow[3]  = _kw(inS[0]),  _kvar(inS[0])
        prow[8],  prow[9]  = _kw(outS[0]), _kvar(outS[0])
    if "b" in phs:
        prow[4],  prow[5]  = _kw(inS[1]),  _kvar(inS[1])
        prow[10], prow[11] = _kw(outS[1]), _kvar(outS[1])
    if "c" in phs:
        prow[6],  prow[7]  = _kw(inS[2]),  _kvar(inS[2])
        prow[12], prow[13] = _kw(outS[2]), _kvar(outS[2])
    pflow_rows.append(prow)
    # losses per phase & totals (kW/kVAr, optional)
    L = inS - outS
    lrow = [b1, b2, None,None, None,None, None,None, None,None]
    if "a" in phs:
        lrow[2] = _round(np.real(L[0])/1000.0, 1)
        lrow[3] = _round(np.imag(L[0])/1000.0, 1)
    if "b" in phs:
        lrow[4] = _round(np.real(L[1])/1000.0, 1)
        lrow[5] = _round(np.imag(L[1])/1000.0, 1)
    if "c" in phs:
        lrow[6] = _round(np.real(L[2])/1000.0, 1)
        lrow[7] = _round(np.imag(L[2])/1000.0, 1)
    lrow[8] = _round(np.real(np.sum(L))/1000.0, 1)
    lrow[9] = _round(np.imag(np.sum(L))/1000.0, 1)
    loss_rows.append(lrow)

# 4) Filter/merge auxiliary buses (distributed loads)
_aux = globals().get('auxiliar_buses', [])
busx_set = set(int(a["busx"]) for a in _aux) if ('has_distributed_load' in globals() and has_distributed_load and _aux) else set()

def _filter_out_busx(rows, pos_from, pos_to):
    return [r for r in rows if int(r[pos_from]) not in busx_set and int(r[pos_to]) not in busx_set]

def _merge_through_busx_currents(crows):
    if not busx_set: return crows
    # in: to==busx, out: from==busx → join on busx → (from_original, to_original)
    cin  = [r[:] for r in crows if int(r[1]) in busx_set]
    cout = [r[:] for r in crows if int(r[0]) in busx_set]
    by_busx_in  = {}
    for r in cin:  by_busx_in.setdefault(int(r[1]), []).append(r)
    by_busx_out = {}
    for r in cout: by_busx_out.setdefault(int(r[0]), []).append(r)
    merged = _filter_out_busx(crows, 0, 1)
    for bx in busx_set:
        for rin in by_busx_in.get(bx, []):
            for rout in by_busx_out.get(bx, []):
                merged.append([rin[0], rout[1]] + rin[2:8] + rout[8:14])
    merged.sort(key=lambda x:(int(x[0]), int(x[1])))
    return merged

def _merge_through_busx_powers(prows):
    if not busx_set: return prows
    pin  = [r[:] for r in prows if int(r[1]) in busx_set]
    pout = [r[:] for r in prows if int(r[0]) in busx_set]
    by_busx_in, by_busx_out = {}, {}
    for r in pin:  by_busx_in.setdefault(int(r[1]), []).append(r)
    for r in pout: by_busx_out.setdefault(int(r[0]), []).append(r)
    merged = _filter_out_busx(prows, 0, 1)
    for bx in busx_set:
        for rin in by_busx_in.get(bx, []):
            for rout in by_busx_out.get(bx, []):
                merged.append([rin[0], rout[1]] + rin[2:8] + rout[8:14])
    merged.sort(key=lambda x:(int(x[0]), int(x[1])))
    return merged

def _merge_through_busx_losses(lrows):
    if not busx_set: return lrows
    aux = [r[:] for r in lrows if (int(r[0]) in busx_set) or (int(r[1]) in busx_set)]
    keep= [r[:] for r in lrows if (int(r[0]) not in busx_set) and (int(r[1]) not in busx_set)]
    # pair m,n with aux[m].to == aux[n].from
    by_to  = {}
    by_from= {}
    for r in aux:
        by_to.setdefault(int(r[1]), []).append(r)
        by_from.setdefault(int(r[0]), []).append(r)
    merged = keep
    for bx in busx_set:
        # phases at busx
        ph = next((a["phases"] for a in auxiliar_buses if int(a["busx"])==bx), "abc") if 'auxiliar_buses' in globals() else "abc"
        for r_in in by_to.get(bx, []):
            for r_out in by_from.get(bx, []):
                newr = [r_in[0], r_out[1], None,None, None,None, None,None, None,None]
                if "a" in ph:
                    newr[2] = _round((r_in[2] or 0)+(r_out[2] or 0), 1)
                    newr[3] = _round((r_in[3] or 0)+(r_out[3] or 0), 1)
                if "b" in ph:
                    newr[4] = _round((r_in[4] or 0)+(r_out[4] or 0), 1)
                    newr[5] = _round((r_in[5] or 0)+(r_out[5] or 0), 1)
                if "c" in ph:
                    newr[6] = _round((r_in[6] or 0)+(r_out[6] or 0), 1)
                    newr[7] = _round((r_in[7] or 0)+(r_out[7] or 0), 1)
                newr[8] = _round((r_in[8] or 0)+(r_out[8] or 0), 1)
                newr[9] = _round((r_in[9] or 0)+(r_out[9] or 0), 1)
                merged.append(newr)
    merged.sort(key=lambda x:(int(x[0]), int(x[1])))
    return merged

# Robust calculation of busx_set
_aux = globals().get("auxiliar_buses", [])
busx_set = {int(a["busx"]) for a in _aux}

# No-op if no auxiliary buses; otherwise always merge
volts_phase_rows = [r for r in volts_phase_rows if int(r[0]) not in busx_set]
volts_pu_rows    = [r for r in volts_pu_rows    if int(r[0]) not in busx_set]
volts_line_rows  = [r for r in volts_line_rows  if int(r[0]) not in busx_set]
cflow_rows = _merge_through_busx_currents(cflow_rows)
pflow_rows = _merge_through_busx_powers(pflow_rows)
loss_rows  = _merge_through_busx_losses(loss_rows)

# 5) Total input power (only substation from row)
sub_from = root_id
pflow_sub = [r for r in pflow_rows if int(r[0])==sub_from]
# kW_in / kVAr_in per phase (0 if missing)
def _sum_col(rows, idx): return sum((r[idx] or 0.0) for r in rows)
total_input_power = [
    _round(_sum_col(pflow_sub, 2), 3), _round(_sum_col(pflow_sub, 4), 3), _round(_sum_col(pflow_sub, 6), 3),
    _round(_sum_col(pflow_sub, 3), 3), _round(_sum_col(pflow_sub, 5), 3), _round(_sum_col(pflow_sub, 7), 3)
]

# 6) Save CSV
with open(_out("volts_phase"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["id","volt_A","deg_A","volt_B","deg_B","volt_C","deg_C"])
    for r in sorted(volts_phase_rows, key=lambda x:int(x[0])): w.writerow(r)

with open(_out("volts_pu"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["id","volt_A","deg_A","volt_B","deg_B","volt_C","deg_C"])  # same header as Julia
    for r in sorted(volts_pu_rows, key=lambda x:int(x[0])): w.writerow(r)

with open(_out("volts_line"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["id","volt_AB","deg_AB","volt_BC","deg_BC","volt_CA","deg_CA"])
    for r in sorted(volts_line_rows, key=lambda x:int(x[0])): w.writerow(r)

with open(_out("current_flow"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["from","to",
        "amp_in_I_ph1","deg_in_I_ph1","amp_in_I_ph2","deg_in_I_ph2","amp_in_I_ph3","deg_in_I_ph3",
        "amp_out_I_ph1","deg_out_I_ph1","amp_out_I_ph2","deg_out_I_ph2","amp_out_I_ph3","deg_out_I_ph3"])
    for r in cflow_rows: w.writerow(r)

with open(_out("power_flow"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["from","to",
        "kW_in_ph1","kVAr_in_ph1","kW_in_ph2","kVAr_in_ph2","kW_in_ph3","kVAr_in_ph3",
        "kW_out_ph1","kVAr_out_ph1","kW_out_ph2","kVAr_out_ph2","kW_out_ph3","kVAr_out_ph3"])
    for r in pflow_rows: w.writerow(r)

# === Power_losses (W / VAr, headers ploss/qloss) ===
def _r1(x): return None if x is None else round(float(x), 1)
def _sum_opt(*vals):
    s = 0.0; seen = False
    for v in vals:
        if v is not None:
            s += float(v); seen = True
    return _r1(s) if seen else None

accP_W  = 0.0   # accumulated active loss [W]
accQ_VAr = 0.0  # accumulated reactive loss [VAr]
# 1) Compute losses per line (before merging auxiliary buses)
rows = []  # [from,to,p1,q1,p2,q2,p3,q3,pt,qt]
id2V   = {int(b["id"]): np.array([b["v_ph1"], b["v_ph2"], b["v_ph3"]], dtype=np.complex128) for b in working_buses}
id2idx = {int(b["id"]): i for i, b in enumerate(working_buses)}

for m, gl in enumerate(gen_lines_mat):
    b1, b2 = int(gl["bus1"]), int(gl["bus2"])
    V1, V2 = id2V[b1], id2V[b2]
    I1     = np.array([lines[m]["ibus1_1"], lines[m]["ibus1_2"], lines[m]["ibus1_3"]], dtype=np.complex128)
    wb2    = working_buses[id2idx[b2]]
    Iout   = np.array([wb2["ibus_1"], wb2["ibus_2"], wb2["ibus_3"]], dtype=np.complex128)
    L      = V1*np.conj(I1) - V2*np.conj(Iout)  # [VA]
    phs    = str(lines[m].get("phases") or "abc").lower()

    p1 = _r1(np.real(L[0])) if "a" in phs else None
    q1 = _r1(np.imag(L[0])) if "a" in phs else None
    p2 = _r1(np.real(L[1])) if "b" in phs else None
    q2 = _r1(np.imag(L[1])) if "b" in phs else None
    p3 = _r1(np.real(L[2])) if "c" in phs else None
    q3 = _r1(np.imag(L[2])) if "c" in phs else None
    pt = _sum_opt(p1, p2, p3)
    qt = _sum_opt(q1, q2, q3)
    rows.append([b1, b2, p1, q1, p2, q2, p3, q3, pt, qt])
    accP_W += float(np.real(L).sum())
    accQ_VAr += float(np.imag(L).sum())

# 2) Merge auxiliary buses (from→busx, busx→to → from→to)
_aux = globals().get("auxiliar_buses", [])
busx_set = {int(a["busx"]) for a in _aux}

if busx_set:
    keep = [r for r in rows if int(r[0]) not in busx_set and int(r[1]) not in busx_set]
    by_to   = {}
    by_from = {}
    for r in rows:
        u, v = int(r[0]), int(r[1])
        if v in busx_set:   by_to.setdefault(v,   []).append(r)
        if u in busx_set:   by_from.setdefault(u, []).append(r)

    merged = []
    for x in busx_set:
        for rin in by_to.get(x, []):      # u -> x
            for rout in by_from.get(x, []):  # x -> w
                u, w = int(rin[0]), int(rout[1])
                # Sum per phase (keep None if absent)
                p1 = _sum_opt(rin[2], rout[2]); q1 = _sum_opt(rin[3], rout[3])
                p2 = _sum_opt(rin[4], rout[4]); q2 = _sum_opt(rin[5], rout[5])
                p3 = _sum_opt(rin[6], rout[6]); q3 = _sum_opt(rin[7], rout[7])
                pt = _sum_opt(p1, p2, p3);     qt = _sum_opt(q1, q2, q3)
                merged.append([u, w, p1, q1, p2, q2, p3, q3, pt, qt])

    rows = keep + merged

# 3) Sort from→to ascending and save (blank for empty phases)
rows.sort(key=lambda r: (int(r[0]), int(r[1])))

with open(_out("power_losses"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["from","to",
                "ploss_ph1","qloss_ph1","ploss_ph2","qloss_ph2","ploss_ph3","qloss_ph3",
                "ploss_totals","qloss_totals"])
    for r in rows:
        w.writerow([r[0], r[1]] + [("" if v is None else v) for v in r[2:]])

with open(_out("total_input_power"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["kW_in_ph1","kW_in_ph2","kW_in_ph3","kVAr_in_ph1","kVAr_in_ph2","kVAr_in_ph3"])
    w.writerow(total_input_power)

# 7) DG summary (if any)
if 'has_distributed_gen' in globals() and has_distributed_gen and generation_register:
    gen_sorted = sorted(generation_register, key=lambda g:int(g["bus"]))
    with open(_out("distributed_generation"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["bus","mode","conn","kw_ph1","kvar_ph1","kw_ph2","kvar_ph2","kw_ph3","kvar_ph3"])
        for g in gen_sorted:
            w.writerow([int(g["bus"]), g["mode"], g["conn"],
                        _round(g["kw_ph1"],3), _round(g["kvar_ph1"],3),
                        _round(g["kw_ph2"],3), _round(g["kvar_ph2"],3),
                        _round(g["kw_ph3"],3), _round(g["kvar_ph3"],3)])

# 8) Print summary
if 'display_summary' in globals() and display_summary:
    print(f"\nmaximum voltage (pu): {ext_max_pu} at bus {ext_max_bus}")
    print(f"minimum voltage (pu): {ext_min_pu} at bus {ext_min_bus}")
    print("Total Input Active Power:  {:.3f} kW".format(sum(total_input_power[:3])))
    print("Total Input Reactive Power:  {:.3f} kVAr".format(sum(total_input_power[3:])))
    print(f"Total Active Power Losses:  {accP_W/1000:.3f} kW")
    print(f"Total Reactive Power Losses: {accQ_VAr/1000:.3f} kVAr")
    print("\nResults in {}".format(output_dir))

# === console display: Distributed Generation table (if exists) ===
def _fmt1to3(x: float) -> str:
    s = f"{float(x):.3f}".rstrip("0")
    return s if s.endswith(".") is False else s + "0"

def _print_dg_table(gen_rows):
    cols = ["bus","mode","conn","kw_ph1","kvar_ph1","kw_ph2","kvar_ph2","kw_ph3","kvar_ph3"]
    # build rows
    rows = []
    for g in gen_rows:
        rows.append([
            int(g["bus"]), str(g["mode"]), str(g["conn"]),
            _fmt1to3(g["kw_ph1"]),  _fmt1to3(g["kvar_ph1"]),
            _fmt1to3(g["kw_ph2"]),  _fmt1to3(g["kvar_ph2"]),
            _fmt1to3(g["kw_ph3"]),  _fmt1to3(g["kvar_ph3"]),
        ])
    # widths
    widths = [max(len(str(c)), *(len(str(r[i])) for r in rows)) for i, c in enumerate(cols)]
    # header
    print(f"\nDistributed Generation: {len(rows)}x{len(cols)} DataFrame")
    hdr = ["Row"] + cols
    w2  = [max(len("Row"), len(str(len(rows))))] + widths
    print("  " + "  ".join(f"{hdr[i]:<{w2[i]}}" for i in range(len(hdr))))
    print("  " + "  ".join("-"*w2[i] for i in range(len(hdr))))
    # rows
    for i, r in enumerate(rows, 1):
        out = [str(i)] + [str(r[j]) for j in range(len(cols))]
        print("  " + "  ".join(f"{out[k]:<{w2[k]}}" for k in range(len(out))))

# Invocation (after saving CSV and before summary output)
if has_distributed_gen and generation_register:
    gen_sorted = sorted(generation_register, key=lambda g: int(g["bus"]))
    _print_dg_table(gen_sorted)