# data_input_py.py
# 목적: SimpleDistributionPowerFlow.jl 의 data_input.jl 동작을
#       Python 3.13 + 표준 라이브러리 + NumPy만으로 "함수 없이" 재현
# 주의: 여기서는 읽기/검증/정규화까지만 수행합니다. 수치 계산은 다른 단계에서 진행.
# 경로/입력/출력 포맷은 줄리아와 동일 규칙 유지.

import os
import csv
import sys
import numpy as np

# ===== 사용자 고정 경로 (요구 5) =====
input_dir  = r"C:\dev\ACPF\BFS\examples\ieee-13"
output_dir = r"C:\dev\ACPF\BFS\results"

# ===== 실행 컨텍스트 (줄리아 read_input_files 인자와 동일 의미) =====
CALLER  = "powerflow"  # "powerflow"일 때 spot_loads/Capacitors/DG까지 읽음
VERBOSE = 1            # 0이 아니면 일부 안내 메시지 출력

# ===== 전역에 해당하는 상태 (줄리아 global들과 동일 키로 유지) =====
bus_coords = []
distributed_gen = []
distributed_loads = []
has_bus_coords = False
has_capacitor = False
has_distributed_gen = False
has_distributed_load = False
has_regulator = False
has_switch = False
has_transformer = False
input_capacitors = []
input_segments = []
line_configurations = []
regulators = []
spot_loads = []
substation = []
switches = []
transformers = []

# ===== 에러 메시지 누적 (줄리아는 함수 반환값으로 err_msg 사용) =====
err_msg = ""

# --------------------------------------------------------------
# [Julia: directory_check(dir, type)] 입력/출력 디렉토리 정규화
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
        # 줄리아 mkpath와 동일: 없으면 생성
        os.makedirs(output_dir, exist_ok=True)
        if not os.path.isabs(output_dir):
            output_dir = os.path.join(os.getcwd(), output_dir)
else:
    output_dir = os.getcwd()

if err_msg:
    print(err_msg)
    sys.exit(1)

# --------------------------------------------------------------
# [Julia: read_input_files(input_directory, caller, verbose)]
#  - CSV 로드(존재/공백 판정)
#  - 컬럼명 정확성 검증(순서 포함)
#  - 단위(ft/mi/m/km) 검증
#  - 중복 검증
#  - 대문자 정규화(config/state/mode/conn 등)
# --------------------------------------------------------------
accepted_units = ["ft", "mi", "m", "km"]

# -- 유틸: CSV 읽기 (함수 없이 구현; 각 파일마다 동일 패턴 사용) --
# 반환: rows(list[dict]), file_err("no file"/"empty file"/"")
# (줄리아 read_file과 동일 의미)
# substation.csv
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
    # 단위 검증
    bad_units = [r for r in input_segments if (r.get("unit") not in accepted_units)]
    if len(bad_units) > 0:
        err_msg = "check for units in 'line_segments.csv' file (only ft, mi, m and km are accepted units)"
        print(err_msg); sys.exit(1)
    # (bus1,bus2) 중복 검증
    pair_set = set()
    for r in input_segments:
        pair = (r.get("bus1"), r.get("bus2"))
        pair_set.add(pair)
    if len(pair_set) != len(input_segments):
        err_msg = "check for duplicated links in 'line_segments.csv' file"
        print(err_msg); sys.exit(1)
    # :config 대문자화(문자열 강제)
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
    # 단위 검증
    bad_units = [r for r in line_configurations if (r.get("unit") not in accepted_units)]
    if len(bad_units) > 0:
        err_msg = "check for units in 'line_configurations.csv' file (only ft, mi, m and km are accepted units)"
        print(err_msg); sys.exit(1)
    # config 대문자화 및 중복 검증
    for r in line_configurations:
        r["config"] = str(r.get("config", "")).upper()
    cfg_codes = [r.get("config") for r in line_configurations]
    if len(set(cfg_codes)) != len(cfg_codes):
        err_msg = "check for duplicated configuration code in 'line_configurations.csv' file"
        print(err_msg); sys.exit(1)

# transformers.csv (선택)
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

# switches.csv (선택)
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

# regulators.csv (선택)
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

# bus_coords.csv (선택)
# bus_coords.csv (선택)
bc_path = os.path.join(input_dir, "bus_coords.csv")
if os.path.isfile(bc_path):
    with open(bc_path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        bc_hdr = list(reader.fieldnames or [])
        bc_rows = list(reader)
    if len(bc_rows) == 0:
        bc_err = "empty file"
    else:
        bc_err = ""
else:
    bc_rows = []
    bc_err = "no file"
bus_coords = bc_rows

# 유효성 검사 후에만 True
valid_bus_coords = False
if bc_err == "":
    expected = ["bus","x","y"]
    if bc_hdr == expected:
        bus_names = [r.get("bus") for r in bus_coords]
        valid_bus_coords = (len(set(bus_names)) == len(bus_names))
    else:
        print("check for column names in 'bus_coords.csv' file. Following without bus coordinates.")
        valid_bus_coords = False
has_bus_coords = valid_bus_coords

# 입력 세그먼트 config 유효성(선로/변압기/스위치/레귤레이터)
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

# distributed_loads.csv (선택)
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
    # 줄리아는 err_msg를 비움
    dl_err = ""

# CALLER가 powerflow일 때 추가 입력
if CALLER == "powerflow":
    # spot_loads.csv (필수)
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

    # capacitors.csv (선택)
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

    # distributed_generation.csv (선택)
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
            # mode 대문자화 + 유효성 검사
            for r in distributed_gen:
                r["mode"] = str(r.get("mode", "")).upper()
            invalid_modes = [r for r in distributed_gen if r.get("mode") not in ("PQ", "PQV", "PI")]
            if len(invalid_modes) > 0:
                err_msg = ("modes of Distributed Generation accepted: PQ (traditional constant watt-var), "
                           "PQV (volt dependant var PQ), PI (constant watt-ampere)")
                print(err_msg); sys.exit(1)
            # bus/conn/mode 결측 제거 알림 (줄리아 dropmissing! 동작을 재현)
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
                # 줄리아는 err_msg 비움
    else:
        if VERBOSE != 0:
            print("no distributed generation")
        # 줄리아는 err_msg 비움

# 최종: 에러 없으면 정상 종료
# 이후 단계에서 이 전역 상태들을 그대로 사용합니다.
if VERBOSE != 0:
    print("\nInput files loaded and validated successfully.")

# === topology_discovery_py.py ===
# 목적: topology_discovery.jl(gridtopology, adjacency_matrix, graph_topology, report_topology)의
#       토폴로지 생성 로직을 함수 없이 순차 블록으로 구현 (Python 3.13 + 표준 라이브러리 + NumPy)
# 전제: 직전 블록(data_input_py.py 변환)에서 아래 전역들이 준비되어 있어야 함.
#   - 입력 데이터: substation, input_segments, line_configurations, transformers, switches,
#                  regulators, distributed_loads, bus_coords
#   - 플래그: has_transformer, has_switch, has_regulator, has_distributed_load, has_bus_coords
#   - 경로: input_dir, output_dir
# 주의: 그래프 시각화(Plots/GraphRecipes)는 미사용. 대신 CSV로 결과를 저장합니다.

import os, csv
import numpy as np
from datetime import datetime

# ==== 사용자 조정 파라미터 (원본 keyword args 대응) ====
CALLER = "powerflow"         # "user" | "powerflow"
GRAPH_TITLE = ""             # (미사용: 이미지 미생성)
MARKER_SIZE = 1.5            # (미사용)
TIMESTAMP = False            # 결과 파일명에 타임스탬프 부여 여부
VERBOSE = 1

# ==== 내부 유틸: 파일명 생성 ====
_ts = ("_" + datetime.now().strftime("%Y%m%d-%H%M")) if TIMESTAMP else ""
def _out(name): return os.path.join(output_dir, f"{name}{_ts}.csv")

# ==== 0) 입력 토폴로지용 버스 목록 생성 (원본: input_buses) ====
# 규칙: line_segments의 bus2 순서 고유 → 이후 bus1 중 아직 없는 것 추가
_input_bus_ids = []
for seg in input_segments:
    b2 = int(seg["bus2"])
    if b2 not in _input_bus_ids:
        _input_bus_ids.append(b2)
for seg in input_segments:
    b1 = int(seg["bus1"])
    if b1 not in _input_bus_ids:
        _input_bus_ids.append(b1)

# 버스 테이블(원본은 DataFrame의 :id 한 열만 사용)
input_buses = [{"id": b} for b in _input_bus_ids]

# ==== 1) 입력 토폴로지 인접행렬(스위치 개방 반영) ====
# 원본 adjacency_matrix(buses, segments) 로직을 그대로 구현
# - buses는 정렬해서 인덱싱
_ids_sorted = sorted([int(r["id"]) for r in input_buses])
_id2idx = {bid: i for i, bid in enumerate(_ids_sorted)}
N_in = len(_ids_sorted)
adj_mat_input = np.zeros((N_in, N_in), dtype=np.int64)

# 기본 간선(방향: bus1 -> bus2)
for seg in input_segments:
    i = _id2idx[int(seg["bus1"])]
    j = _id2idx[int(seg["bus2"])]
    adj_mat_input[i, j] = 1

# 스위치 구성인 세그먼트는 상태에 따라 간선 제거
if has_switch:
    line_cfg_codes = set([str(r["config"]) for r in line_configurations])
    sw_cfg_codes = set([str(r["config"]) for r in switches])
    for seg in input_segments:
        cfg = str(seg["config"])
        if (cfg not in line_cfg_codes) and (cfg in sw_cfg_codes):
            # 이 세그먼트는 스위치 장치임
            # 해당 스위치 상태 조회
            st = None
            for s in switches:
                if str(s["config"]) == cfg:
                    st = str(s["state"]).upper()
                    break
            if st is not None and st != "CLOSED":
                i = _id2idx[int(seg["bus1"])]
                j = _id2idx[int(seg["bus2"])]
                adj_mat_input[i, j] = 0

# ==== 2) 워킹 토폴로지 구성(스위치 반영, 단절 세그먼트 제거) ====
# working_segments = input_segments + check(=0) 열 → open 스위치는 check=1로 표시 후 제거
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

# ==== 3) 루트로부터 연결된 버스만 남기기 ====
# 시드: 서브스테이션 버스(:bus → :id)
if not substation:
    raise RuntimeError("substation.csv is required and must have at least one row.")
root_bus = int(substation[0]["bus"])
working_buses_ids = [root_bus]
increase_monitor = 1

# 원본과 동일하게: working_buses[n]를 확장하는 형태로 순차 추가 후 확장 없으면 break
for _ in range(len(working_segments)):
    # 현재 인덱스 n에 해당하는 루트부터 출발 엣지 검색
    n_idx = len(working_buses_ids) - 1  # 마지막 추가된 버스 기준
    # 원본은 1..nrow(working_segments) 루프에서 working_buses[n,:id] 사용
    # 여기서는 같은 효과를 내도록 현재까지 확보된 모든 bus에서 확장
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

# ==== 4) 스위치로 인한 방향성 보정(부족 버스가 있으면 일부 세그먼트 방향 스왑) ====
if has_switch and (len(working_buses_ids) != len(_input_bus_ids)):
    buses_diff = len(_input_bus_ids) - len(working_buses_ids)
    for _ in range(buses_diff):
        # bus2는 연결되었으나 bus1이 아직 미포함인 세그먼트 후보
        tmp = []
        for seg in working_segments:
            b1 = int(seg["bus1"]); b2 = int(seg["bus2"])
            if (b2 in working_buses_ids) and (b1 not in working_buses_ids):
                tmp.append({"bus1": b1, "bus2": b2})
        if len(tmp) > 0:
            pick = tmp[0]
            # 부족 버스 추가
            working_buses_ids.append(int(pick["bus1"]))
            # 대응 세그먼트 방향 스왑
            for seg in working_segments:
                if int(seg["bus1"]) == pick["bus1"] and int(seg["bus2"]) == pick["bus2"]:
                    seg["bus1"], seg["bus2"] = seg["bus2"], seg["bus1"]

# ==== 5) 워킹 세그먼트 중 루트에서 도달 불가한 세그먼트 제거 ====
for seg in working_segments:
    if int(seg["bus1"]) not in working_buses_ids:
        seg["_check"] = 1
working_segments = [r for r in working_segments if r["_check"] == 0]
for seg in working_segments:
    if "_check" in seg: del seg["_check"]

# ==== 6) 루프 존재 여부 검사 ====
if (len(working_segments) - len(working_buses_ids) + 1) > 0:
    msg = f"Topology has a loop, this version only works with radial topologies. See result in {output_dir}."
    print(msg)
    raise RuntimeError(msg)

# ==== 7) 분포부하 처리: 보조 버스 삽입(각 구간을 1/2씩 나눔) ====
auxiliar_buses = []   # [{bus1, bus2, busx}]
if has_distributed_load and distributed_loads:
    # 길이 실수화
    for seg in working_segments:
        seg["length"] = float(seg["length"])

    # 분포부하에 해당하는 세그먼트를 수집
    dist_load_segments = []
    for dl in distributed_loads:
        b1 = int(dl["bus1"]); b2 = int(dl["bus2"])
        for seg in working_segments:
            if int(seg["bus1"]) == b1 and int(seg["bus2"]) == b2:
                dist_load_segments.append(dict(seg))  # 카피

    # 수집된 세그먼트들을 원본 워킹 세그먼트에서 제거
    def _is_same(a, b):
        return int(a["bus1"]) == int(b["bus1"]) and int(a["bus2"]) == int(b["bus2"])
    working_segments = [s for s in working_segments if all(not _is_same(s, d) for d in dist_load_segments)]

    # 중간 버스 생성하여 2개 세그먼트로 분할
    next_bus_id = max(working_buses_ids) + 1 if working_buses_ids else 1
    for dseg in dist_load_segments:
        start_bus = int(dseg["bus1"])
        end_bus   = int(dseg["bus2"])
        unit = dseg["unit"]; conf = dseg["config"]
        L = float(dseg["length"]); L1 = L * 0.5; L2 = L * 0.5
        # 앞쪽 절반
        working_segments.append({"bus1": start_bus, "bus2": next_bus_id,
                                 "length": L1, "unit": unit, "config": conf})
        # 뒤쪽 절반
        working_segments.append({"bus1": next_bus_id, "bus2": end_bus,
                                 "length": L2, "unit": unit, "config": conf})
        auxiliar_buses.append({"bus1": start_bus, "bus2": end_bus, "busx": next_bus_id})
        working_buses_ids.append(next_bus_id)
        next_bus_id += 1

    # 좌표가 있으면 중점 좌표 생성
    if has_bus_coords and bus_coords:
        known = set(int(r["bus"]) for r in bus_coords)
        no_coords = [b for b in working_buses_ids if b not in known]
        for bx in no_coords:
            # pre: ... -> bx, post: bx -> ...
            pre_seg  = next((s for s in working_segments if int(s["bus2"]) == bx), None)
            post_seg = next((s for s in working_segments if int(s["bus1"]) == bx), None)
            if pre_seg and post_seg:
                pre = int(pre_seg["bus1"]); post = int(post_seg["bus2"])
                pre_c  = next(r for r in bus_coords if int(r["bus"]) == pre)
                post_c = next(r for r in bus_coords if int(r["bus"]) == post)
                if float(pre_c["x"]) == float(post_c["x"]):
                    new_x = float(pre_c["x"])
                    new_y = 0.5 * (float(pre_c["y"]) + float(post_c["y"]))
                elif float(pre_c["y"]) == float(post_c["y"]):
                    new_y = float(pre_c["y"])
                    new_x = 0.5 * (float(pre_c["x"]) + float(post_c["x"]))
                else:
                    new_x = 0.5 * (float(pre_c["x"]) + float(post_c["x"]))
                    new_y = 0.5 * (float(pre_c["y"]) + float(post_c["y"]))
                bus_coords.append({"bus": bx, "x": new_x, "y": new_y})

# ==== 8) 워킹 토폴로지 인접행렬 산출 ====
_ids_sorted_w = sorted(list(set(int(b) for b in working_buses_ids)))
_id2idx_w = {bid: i for i, bid in enumerate(_ids_sorted_w)}
N_w = len(_ids_sorted_w)
adj_mat_work = np.zeros((N_w, N_w), dtype=np.int64)

for seg in working_segments:
    i = _id2idx_w[int(seg["bus1"])]
    j = _id2idx_w[int(seg["bus2"])]
    adj_mat_work[i, j] = 1

# 스위치 세그먼트(장치형) 개방 상태면 0
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

# ==== 9) (옵션) 줄리아 결과 CSV와 비교 ====
# 주: 원본은 PNG만 저장하므로 직접 비교 파일이 없을 수 있습니다.
#    baseline 파일이 존재하면 allclose로 비교하고, 없으면 건너뜁니다.
def _try_compare(a_path, b_path, name):
    try:
        if os.path.isfile(a_path) and os.path.isfile(b_path):
            A = np.loadtxt(a_path, delimiter=',')
            B = np.loadtxt(b_path, delimiter=',')
            ok = np.allclose(A, B, atol=1e-8, rtol=1e-6, equal_nan=True)
            print(f"[COMPARE] {name}: {'OK' if ok else 'MISMATCH'}")
    except Exception as e:
        print(f"[COMPARE] {name}: skipped ({e})")

# 예시: 사용자가 baseline을 제공했다면 아래 경로를 수정해서 사용
# _try_compare(r"C:\path\to\baseline\sdpf_input_topology_adj.csv",  _out("sdpf_input_topology_adj"),  "input_adj")
# _try_compare(r"C:\path\to\baseline\sdpf_working_topology_adj.csv", _out("sdpf_working_topology_adj"), "working_adj")

# ==== 종료 메시지 (원본 caller별 리턴 모드 대응) ====
if CALLER == "user":
    print(f"Execution finished, see results in {output_dir}.")

# === data_preparation_py.py ===
# 목적: data_preparation.jl 의 data_preparation() + working_lines() 동작을
#       Python 3.13 + NumPy만으로 "함수 없이" 순차 구현
# 전제: 이전 단계에서 다음 전역이 준비됨:
#  - substation, working_segments, line_configurations, transformers, switches,
#    regulators, spot_loads, distributed_loads, input_capacitors, distributed_gen
#  - has_transformer, has_switch, has_regulator, has_distributed_load, has_distributed_gen
#  - input_dir, output_dir
# 주의: pandas 미사용, 리스트+dict로 DataFrame 대체. dtype은 float64/complex128 고정.

import math, os, csv
import numpy as np

# ---------- 공통 유틸 ----------
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

# ---------- (안전) working_buses / adj_mat 확보 ----------
try:
    working_buses
except NameError:
    working_buses = []

if not working_buses:
    # 토폴로지 단계에서 리스트를 만들지 않았다면 세그먼트로부터 생성
    _wb_ids = _ids_from_segments(working_segments)
    working_buses = [{"id": i} for i in _wb_ids]

try:
    adj_mat  # 토폴로지에서 생성되었는지 확인
except NameError:
    # 없으면 working_segments로부터 생성
    _wb_ids = sorted([b["id"] for b in working_buses])
    _idx = {bid: i for i, bid in enumerate(_wb_ids)}
    N = len(_wb_ids)
    adj_mat = np.zeros((N, N), dtype=np.int64)
    for seg in working_segments:
        i = _idx[_to_int(seg["bus1"])]
        j = _idx[_to_int(seg["bus2"])]
        adj_mat[i, j] = 1

# ---------- working_lines() 변환 블록 ----------
# lines = working_segments + :type (1=line, 2=transformer, 3=switch, 4=regulator)
lines = [dict(r) for r in working_segments]
for r in lines:
    r["type"] = 0  # 초기값

# 타입 판정
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

# working_buses에 :type, :number 컬럼 추가
for b in working_buses:
    b["type"] = 0   # 1=sub,2=bif,3=intermediate,4=next-to-end,5=end
    b["number"] = 0

# out-degree(=downward_buses) 계산
# adj_mat 행합(축=1)
downward_buses = np.sum(adj_mat, axis=1).astype(np.int64)  # shape (N,)
# id→행인덱스 매핑
wb_ids_sorted = sorted([b["id"] for b in working_buses])
wb_id2idx = {bid: i for i, bid in enumerate(wb_ids_sorted)}

# 1차 마킹: 종단/분기
for b in working_buses:
    deg = downward_buses[wb_id2idx[b["id"]]]
    if deg == 0:
        b["type"] = 5  # ending
    elif deg > 1:
        b["type"] = 2  # bifurcation

# 2차 마킹: out-degree=1 → 다음 버스가 종단이면 4, 아니면 3
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

# 변전소(서브스테이션) 버스 지정
root_bus = _to_int(substation[0]["bus"])
for b in working_buses:
    if b["id"] == root_bus:
        b["type"] = 1

# type 오름차순 정렬
working_buses.sort(key=lambda x: x["type"])

# 초기 번호 부여 (type ∈ {1,4,5})
for idx, b in enumerate(working_buses, start=1):
    if b["type"] in (1, 4, 5):
        b["number"] = idx

# 통계 및 역번호 시작점 k
initial_buses     = sum(1 for b in working_buses if b["type"] == 1)
bifurcation_buses = sum(1 for b in working_buses if b["type"] == 2)
interm_buses      = sum(1 for b in working_buses if b["type"] == 3)
next_to_end_buses = sum(1 for b in working_buses if b["type"] == 4)
end_buses         = sum(1 for b in working_buses if b["type"] == 5)
k = len(working_buses) - end_buses - next_to_end_buses

# 중간/분기 버스 번호 매기기
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

# 남은 미번호 버스 처리(선행자 번호+1)
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
# 번호 기준 오름차순
working_buses.sort(key=lambda x: x["number"])

# 변압기 하류 마킹
for b in working_buses:
    b["trf"] = None

if has_transformer and transformers:
    for t in transformers:
        for seg in working_segments:
            if str(seg["config"]) == str(t["config"]):
                child = _first_row(working_buses, lambda x: x["id"] == _to_int(seg["bus2"]))
                if child:
                    child["trf"] = t["config"]

    # 전파
    for b in working_buses:
        if (b["trf"] is not None) and (b["type"] != 5):
            for seg in working_segments:
                if _to_int(seg["bus1"]) == b["id"]:
                    child = _first_row(working_buses, lambda x: x["id"] == _to_int(seg["bus2"]))
                    if child:
                        child["trf"] = b["trf"]

# ---------- data_preparation() 변환 블록 ----------
err_msg = ""

# 회전상수 및 변환행렬
as_ = np.exp(1j * np.deg2rad(120.0))            # = 1 * exp(j*120°)
As  = np.array([[1, 1, 1],
                [1, as_**2, as_],
                [1, as_, as_**2]], dtype=np.complex128)  # 참고: 원본 As는 후속 코드에서 직접 사용되지 않음

# 기준 전압(상간/상전) 및 변환행렬 D
ell = _to_float(substation[0]["kv"]) * 1000.0                    # [V_LL]
eln = ell / math.sqrt(3.0)                                       # [V_LN]
ELN = np.array([eln,
                eln * np.exp(-1j * np.deg2rad(120.0)),
                eln * np.exp( 1j * np.deg2rad(120.0))], dtype=np.complex128)
D = np.array([[ 1.0, -1.0,  0.0],
              [ 0.0,  1.0, -1.0],
              [-1.0,  0.0,  1.0]], dtype=np.float64)
ELL = D @ ELN.astype(np.complex128)

# working_buses에 v_base 추가(변압기 2차측이면 kv_low/√3, 아니면 eln)
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

# line_configs 구성(저항/리액턴스→복소 임피던스, B계수는 j*값)
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
        # baa..bcc는 원본에서 1im을 곱해 순수허수로 저장 후, 길이*단위환산*(1e-6)을 곱합니다.
        "baa": 1j * _to_float(cfg["baa"]),
        "bab": 1j * _to_float(cfg["bab"]),
        "bac": 1j * _to_float(cfg["bac"]),
        "bbb": 1j * _to_float(cfg["bbb"]),
        "bbc": 1j * _to_float(cfg["bbc"]),
        "bcc": 1j * _to_float(cfg["bcc"]),
    }

line_configs = [_cfg_row(r) for r in line_configurations]

# lines에 phases/Zxx/Bxx 열 추가(초기 None)
for r in lines:
    r["phases"] = None
    for key in ("Zaa","Zab","Zac","Zbb","Zbc","Zcc","Baa","Bab","Bac","Bbb","Bbc","Bcc"):
        r[key] = None

# 변압기 Zt 추가(Ω)
if has_transformer and transformers:
    for t in transformers:
        kv_low = _to_float(t["kv_low"])
        kva    = _to_float(t["kva"])
        zpu    = _to_float(t["rpu"]) + 1j * _to_float(t["xpu"])
        t["Zt"] = (kv_low**2 / kva) * zpu * 1000.0  # [Ω], complex128

# 선로별 Z/Y 및 phases 판정
for r in lines:
    t = _to_int(r["type"])
    if t == 1:  # line
        cfg = _first_row(line_configs, lambda c: c["config"] == str(r["config"]))
        if cfg is None:
            continue
        # 단위 환산계수(라인 세그먼트 단위 → 라인 configuration 단위)
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
        # µS → S 로 1e-6 곱함. cfg["baa"] 등은 이미 j*값 형태임
        r["Baa"] = cfg["baa"] * L * factor * 1e-6
        r["Bab"] = cfg["bab"] * L * factor * 1e-6
        r["Bac"] = cfg["bac"] * L * factor * 1e-6
        r["Bbb"] = cfg["bbb"] * L * factor * 1e-6
        r["Bbc"] = cfg["bbc"] * L * factor * 1e-6
        r["Bcc"] = cfg["bcc"] * L * factor * 1e-6
        # phases 판정
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

# 일반화 선로행렬 gen_lines_mat (Kersting 4ed ch.6) 생성
gen_lines_mat = []
U = np.eye(3, dtype=np.complex128)
z_line = np.zeros((3,3), dtype=np.complex128)
y_line = np.zeros((3,3), dtype=np.complex128)

for r in lines:
    # 대칭행렬 구성 (Zxx, Bxx는 위에서 설정)
    z_line[0,0] = r["Zaa"]; z_line[0,1] = r["Zab"]; z_line[0,2] = r["Zac"]
    z_line[1,0] = r["Zab"]; z_line[1,1] = r["Zbb"]; z_line[1,2] = r["Zbc"]
    z_line[2,0] = r["Zac"]; z_line[2,1] = r["Zbc"]; z_line[2,2] = r["Zcc"]

    y_line[0,0] = r["Baa"]; y_line[0,1] = r["Bab"]; y_line[0,2] = r["Bac"]
    y_line[1,0] = r["Bab"]; y_line[1,1] = r["Bbb"]; y_line[1,2] = r["Bbc"]
    y_line[2,0] = r["Bac"]; y_line[2,1] = r["Bbc"]; y_line[2,2] = r["Bcc"]

    t = _to_int(r["type"])
    # 기본 a,b,c,d
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
            a=b=c=d=A=B = np.zeros((3,3), dtype=np.complex128)  # 방어

    # gen_lines_mat 행 저장
    rec = {
        "bus1": _to_int(r["bus1"]),
        "bus2": _to_int(r["bus2"]),
    }
    # 소문자 a,b,c,d 와 대문자 A,B (각 3x3를 낱개 요소로)
    for name, M in (("a",a),("b",b),("c",c),("d",d),("A",A),("B",B)):
        for i in range(3):
            for j in range(3):
                rec[f"{name}_{i+1}_{j+1}"] = np.complex128(M[i,j])
    gen_lines_mat.append(rec)

# ---------- 부하 테이블 loads 구성 ----------
loads = []  # 각 원소: {"bus":int, "conn":str, "type":str, "ph_1":complex, "ph_2":complex, "ph_3":complex}

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

# distributed loads (보조버스 busx에 부착)
if has_distributed_load and distributed_loads:
    # bus1,bus2 → busx 매핑 준비
    _busx_map = {}
    for a in globals().get("auxiliar_buses", []):
        _busx_map[(int(a["bus1"]), int(a["bus2"]))] = int(a["busx"])
        _busx_map[(int(a["bus2"]), int(a["bus1"]))] = int(a["busx"])  # 양방향 키 허용

    for r in distributed_loads:
        b1 = _to_int(r["bus1"]); b2 = _to_int(r["bus2"])
        target_bus = _busx_map.get((b1, b2), None)

        # 안전 폴백: 매핑을 못 찾으면 기존 방식으로 시작버스에 부착
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


# capacitors → 음의 무효전력 (Y, Z)
for r in input_capacitors:
    b = _to_int(r["bus"])
    if b in wb_id_set:
        s1 = -1j * _to_float(r["kvar_ph1"]) * 1000.0
        s2 = -1j * _to_float(r["kvar_ph2"]) * 1000.0
        s3 = -1j * _to_float(r["kvar_ph3"]) * 1000.0
        loads.append({"bus": b, "conn": "Y", "type": "Z",
                      "ph_1": np.complex128(s1), "ph_2": np.complex128(s2), "ph_3": np.complex128(s3)})

# ---------- 분산발전(DG) 식별/적재 ----------
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
    # 결측 제거
    _valid = []
    for r in pq:
        if not (_is_missing(r.get("kw_set")) or _is_missing(r.get("kvar_set"))):
            _valid.append(r)
        else:
            print("PQ distributed generation with missing values, it will be ignored.")
    pq = [r for r in _valid if _to_int(r.get("bus")) in wb_id_set]  # 존재 버스만
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
        # 상태 추적 컬럼들(0으로 초기화)
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

    # DG 종류가 하나도 없으면 전체 플래그 false
    if not (has_pq_distributed_gen or has_pqv_distributed_gen or has_pi_distributed_gen):
        has_distributed_gen = False

# loads에 상수 k_1~k_3 추가 (Z:I 변환상수)
for ld in loads:
    ld["k_1"] = None; ld["k_2"] = None; ld["k_3"] = None
    if ld["type"] in ("Z","I"):
        # 버스 기준전압(v_base) 결정 (Y: V_LN, Δ: √3·V_LN)
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
            # k = |S| / V_nom  (전류 크기[A])
            for kname, sph in (("k_1","ph_1"),("k_2","ph_2"),("k_3","ph_3")):
                S = ld[sph]
                if S == 0 or S == 0.0:
                    ld[kname] = 0.0
                else:
                    ld[kname] = np.float64(abs(S) / vb)

# working_buses에 process(Int8), phases, v_ph1~3, ibus_1~3 추가
for b in working_buses:
    b["process"] = np.int8(0)
    b["phases"]  = None
    b["v_ph1"] = np.complex128(0+0j)
    b["v_ph2"] = np.complex128(0+0j)
    b["v_ph3"] = np.complex128(0+0j)
# number 내림차순 정렬(원본: rev=true)
working_buses.sort(key=lambda x: x["number"], reverse=True)
for b in working_buses:
    b["ibus_1"] = np.complex128(0+0j)
    b["ibus_2"] = np.complex128(0+0j)
    b["ibus_3"] = np.complex128(0+0j)

# lines에 ibus1_1~3 초기화
for r in lines:
    r["ibus1_1"] = np.complex128(0+0j)
    r["ibus1_2"] = np.complex128(0+0j)
    r["ibus1_3"] = np.complex128(0+0j)

# 임시 버스 전압·전류 컨테이너(3상)
Vbus1 = np.zeros(3, dtype=np.complex128)
Vbus2 = np.zeros(3, dtype=np.complex128)
Ibus1 = np.zeros(3, dtype=np.complex128)
Ibus2 = np.zeros(3, dtype=np.complex128)



# === power_flow_py.py ===
# 목적: power_flow.jl 의 외부 반복(PQV/PI DG 보정) + 최종 결과 CSV 산출을
#       "단일 스크립트" 형태로 구현 (Python 3.13 + NumPy)
# 전제: 이전 단계(data_input → topology_discovery → data_preparation → sweep_procedures) 코드가
#       같은 파일 상단에 이미 존재하여 전역 상태가 준비되어 있어야 합니다.

import os, csv, math
import numpy as np

# ---------- 실행 파라미터(상위에서 지정 없을 때 기본값) ----------
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

# ---------- 타임스탬프 스UFFix ----------
from datetime import datetime
_ts = ("_" + datetime.now().strftime("%Y%m%d-%H%M")) if timestamp else ""
def _out(name): return os.path.join(output_dir, f"{name}{_ts}.csv")

# ---------- 도우미 ----------
def _deg(z):  # 복소수 각도[deg]
    return np.rad2deg(np.angle(z))
def _safe_rel_diff(old: np.ndarray, new: np.ndarray):
    old = np.asarray(old, dtype=np.float64)
    new = np.asarray(new, dtype=np.float64)
    denom = np.where(np.abs(new) > 0, np.abs(new), 1.0)
    return np.max(np.abs((old - new) / denom))
def _dump_complex_as_mag_deg(writer, zs):  # (mag,deg) 반복기록
    for z in zs:
        writer.writerow([np.abs(z), _deg(z)])

# =====================================================================
# 외부 반복 루프 (PQV/PI DG만 보정; PQ는 data_preparation에서 이미 고정 주입)
# =====================================================================
outer_iteration = 0
max_diff = 1.0
inner_iteration = 0  # 마지막 외부 라운드의 inner 반복수

while max_diff > tolerance:
    # ---------------------------
    # 1) FBS "내부 반복"을 수렴까지 실행
    #    (sweep_procedures 블록의 while 루프를 재사용)
    # ---------------------------
    # 초기화
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

        # 종단 버스: 부하전류
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

        # 비종단: 자식→부모 환산 + 국소부하 합산
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

            # 국소 부하 합산
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

        # 3) 서브스테이션 전압 오차 평가
        _sub_id = int(substation[0]["bus"])
        _sub = next(b for b in working_buses if int(b["id"]) == _sub_id)
        sub_phase = np.array([_sub["v_ph1"], _sub["v_ph2"], _sub["v_ph3"]], dtype=np.complex128)
        sub_line = D @ sub_phase
        errs = np.abs((np.abs(ELL) - np.abs(sub_line)) / np.abs(ELL))
        max_error = float(np.max(errs))

        if iter_number == max_iterations:
            print(f"Program halted: maximum number of forward-backward iterations reached ({max_iterations})")
            break

    # 이 외부 라운드의 내부 반복수 기록
    inner_iteration = iter_number

    # ---------------------------
    # 2) DG 보정: PQV, PI
    # ---------------------------
    if not has_distributed_gen:
        max_diff = 0.0
        outer_iteration += 1
        break

    # (1) PQV
    if has_pqv_distributed_gen and len(pqv_distributed_gen) > 0:
        # 현재 전압(절대값) 수집
        id2V = {int(wb["id"]): (abs(wb["v_ph1"]), abs(wb["v_ph2"]), abs(wb["v_ph3"])) for wb in working_buses}

        for rec in pqv_distributed_gen:
            bus = int(rec["bus"])
            v_old = np.array([rec["v_ph1"], rec["v_ph2"], rec["v_ph3"]], dtype=np.float64)
            v_new = np.array(id2V[bus], dtype=np.float64)

            v_set = float(rec["kv_set"]) * 1000.0 / math.sqrt(3.0)
            xd    = float(rec["xd"])
            p_ph  = float(rec["kw_set"]) * 1000.0 / 3.0

            # 식: var = sqrt((v_set*|v|/xd)^2 - p_ph^2) - |v|^2/xd  (상별)
            # 수치 안전: 루트 음수면 0으로 취급(원본은 직접 sqrt; 음수 발생 시 도메인 에러 방지 목적)
            var = []
            for k in range(3):
                term = (v_set * v_new[k] / xd)**2 - p_ph**2
                term = term if term > 0 else 0.0
                var_k = math.sqrt(term) - (v_new[k]**2) / xd
                var.append(var_k)
            var = np.asarray(var, dtype=np.float64)

            # 한계치 클램프
            qmin = float(rec["kvar_min"]) * 1000.0 / 3.0
            qmax = float(rec["kvar_max"]) * 1000.0 / 3.0
            var = np.clip(var, qmin, qmax)

            # loads에서 기존 DG행 제거(type == "PQV" & bus == rec["bus"])
            loads = [ld for ld in loads if not (int(ld["bus"]) == bus and ld["type"] == "PQV")]
            # 새 DG 행 삽입 (상별 S = P + jQ, 주입은 음수표기)
            s_ph = p_ph + 1j*var
            loads.append({"bus": bus, "conn": rec["conn"], "type": "PQV",
                          "ph_1": -np.complex128(s_ph[0]),
                          "ph_2": -np.complex128(s_ph[1]),
                          "ph_3": -np.complex128(s_ph[2])})

            # 변화율 기록
            max_volt_diff = _safe_rel_diff(v_old, v_new)
            rec["v_ph1"], rec["v_ph2"], rec["v_ph3"] = float(v_new[0]), float(v_new[1]), float(v_new[2])
            rec["max_diff"] = float(max_volt_diff)
            rec["w_ph1"] = p_ph; rec["w_ph2"] = p_ph; rec["w_ph3"] = p_ph
            rec["var_ph1"] = float(var[0]); rec["var_ph2"] = float(var[1]); rec["var_ph3"] = float(var[2])

            # generation_register 업데이트(버스별 단일 행 유지)
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

            # 결선별 전압/전류 기준
            if rec["conn"] == "Y":
                v_ph = v_new.copy()
                i_ph = i_set
            else:  # "D"
                # 선간전압 절대값
                v_ph = np.array([abs(v_bus[0]-v_bus[1]),
                                 abs(v_bus[1]-v_bus[2]),
                                 abs(v_bus[2]-v_bus[0])], dtype=np.float64)
                i_ph = i_set / math.sqrt(3.0)

            # q_ph = sqrt( (i_ph*v)^2 - p_ph^2 ), 도메인 음수→qmin
            q_ph = []
            for k in range(3):
                term = (i_ph * v_ph[k])**2 - p_ph**2
                qk = math.sqrt(term) if term >= 0 else qmin
                q_ph.append(qk)
            q_ph = np.clip(np.array(q_ph, dtype=np.float64), qmin, qmax)

            # loads 교체(type=="PI")
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

    # (3) 종료 판단
    if len(generation_register) > 0:
        max_diff = max(float(g["max_diff"]) for g in generation_register)
    else:
        max_diff = 0.0

    outer_iteration += 1

# === results_py (ported from print_results.jl) ===
# 요구 전역: working_buses, lines, gen_lines_mat, substation, D, ELL, output_dir,
#            has_distributed_load, auxiliar_buses (있으면), has_distributed_gen, generation_register,
#            display_summary(bool), timestamp(bool)
import os, csv, math
import numpy as np
from datetime import datetime

def _deg(z): return float(np.rad2deg(np.angle(z)))
def _round(x, d): return (None if x is None else (round(float(x), d)))
def _out(name):
    ts = "_" + datetime.now().strftime("%Y%m%d-%H%M") if 'timestamp' in globals() and timestamp else ""
    return os.path.join(output_dir, f"{name}{ts}.csv")

# 0) 각 버스의 phases 채우기 (루트는 'abc')
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

# 보조버스가 있다면 phases 보강
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

# 2) Volts line (LL) — 선택적 채움
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
# in_I_ph* from lines[*]["ibus1_*"]; out_I_ph* = working_buses at 'to' (JL 방식)
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
    # current report (선택적 채움)
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
    # power report (kW/kVAr, 선택적 채움)
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
    # losses per phase & totals (kW/kVAr, 선택적 채움)
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

# 4) 보조버스 필터/병합 (distributed loads)
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

# busx_set 강건 산출
_aux = globals().get("auxiliar_buses", [])
busx_set = {int(a["busx"]) for a in _aux}

# (보조버스가 없다면 noop이지만, 있으면 항상 병합 수행)
volts_phase_rows = [r for r in volts_phase_rows if int(r[0]) not in busx_set]
volts_pu_rows    = [r for r in volts_pu_rows    if int(r[0]) not in busx_set]
volts_line_rows  = [r for r in volts_line_rows  if int(r[0]) not in busx_set]
cflow_rows = _merge_through_busx_currents(cflow_rows)
pflow_rows = _merge_through_busx_powers(pflow_rows)
loss_rows  = _merge_through_busx_losses(loss_rows)

# 5) 총 입력 전력 (서브스테이션 from 행만)
sub_from = root_id
pflow_sub = [r for r in pflow_rows if int(r[0])==sub_from]
# kW_in / kVAr_in per phase (없으면 0)
def _sum_col(rows, idx): return sum((r[idx] or 0.0) for r in rows)
total_input_power = [
    _round(_sum_col(pflow_sub, 2), 3), _round(_sum_col(pflow_sub, 4), 3), _round(_sum_col(pflow_sub, 6), 3),
    _round(_sum_col(pflow_sub, 3), 3), _round(_sum_col(pflow_sub, 5), 3), _round(_sum_col(pflow_sub, 7), 3)
]

# 6) CSV 저장
with open(_out("volts_phase"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["id","volt_A","deg_A","volt_B","deg_B","volt_C","deg_C"])
    for r in sorted(volts_phase_rows, key=lambda x:int(x[0])): w.writerow(r)

with open(_out("volts_pu"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["id","volt_A","deg_A","volt_B","deg_B","volt_C","deg_C"])  # jl과 동일 헤더 구조
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

# === sdpf_power_losses (줄리아 규격: W / VAr, 헤더 ploss/qloss) ===
def _r1(x): return None if x is None else round(float(x), 1)
def _sum_opt(*vals):
    s = 0.0; seen = False
    for v in vals:
        if v is not None:
            s += float(v); seen = True
    return _r1(s) if seen else None

accP_W  = 0.0   # 누적 유효손실 [W]
accQ_VAr = 0.0  # 누적 무효손실 [VAr]
# 1) 선로별 손실 계산 (보조버스 병합 전)
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

# 2) 보조버스 병합 (from→busx, busx→to → from→to)
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
                # 상별 합산(없으면 None 유지)
                p1 = _sum_opt(rin[2], rout[2]); q1 = _sum_opt(rin[3], rout[3])
                p2 = _sum_opt(rin[4], rout[4]); q2 = _sum_opt(rin[5], rout[5])
                p3 = _sum_opt(rin[6], rout[6]); q3 = _sum_opt(rin[7], rout[7])
                pt = _sum_opt(p1, p2, p3);     qt = _sum_opt(q1, q2, q3)
                merged.append([u, w, p1, q1, p2, q2, p3, q3, pt, qt])

    rows = keep + merged

# 3) from→to 오름차순 정렬 + 저장(빈 상은 공란)
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

# 7) DG 요약 (있을 때만)
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

# 8) 요약 출력
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

# 호출 (CSV 저장 후, 요약 출력 전에)
if has_distributed_gen and generation_register:
    gen_sorted = sorted(generation_register, key=lambda g: int(g["bus"]))
    _print_dg_table(gen_sorted)