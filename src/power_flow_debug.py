# -*- coding: utf-8 -*-
import sys
import pandas as pd
import numpy as np  # (필요할 수 있어 추가)
import shared_state as ST
import topology_discovery   # gridtopology(...)
import data_preparation     # data_preparation()
import sweep_procedures     # forward_backward_sweep(tol, max_iter) -> (err_msg, inner_iter)
import print_results        # results(display_summary, timestamp)

# --------------------------------------------------------------------
# 사용자 설정(필요시 수정)
# --------------------------------------------------------------------
input_dir = r"C:\dev\ACPF\BFS\examples\ieee-13"
output_dir = ""

tolerance = 1e-6
max_iterations = 30

display_summary = True
timestamp = False
display_topology = False
save_topology = False
graph_title = "ieee-13"
marker_size = 1.5
verbose = 1

# --------------------------------------------------------------------
# (옵션) 공백/결측 방지: DG 플래그/테이블이 없을 때 기본값 세팅
# --------------------------------------------------------------------
if not hasattr(ST.state, "has_distributed_gen"):
    ST.state.has_distributed_gen = False
if not hasattr(ST.state, "has_pqv_distributed_gen"):
    ST.state.has_pqv_distributed_gen = False
if not hasattr(ST.state, "has_pi_distributed_gen"):
    ST.state.has_pi_distributed_gen = False
if not hasattr(ST.state, "generation_register") or ST.state.generation_register is None:
    ST.state.generation_register = pd.DataFrame(columns=[
        "bus","mode","conn","kw_ph1","kvar_ph1","kw_ph2","kvar_ph2","kw_ph3","kvar_ph3","max_diff"
    ])

# --------------------------------------------------------------------
# 1) Topology discovery
# --------------------------------------------------------------------
err_msg = topology_discovery.gridtopology(
    caller="powerflow",
    input_dir=input_dir,
    output_dir=output_dir,
    timestamp=timestamp,
    save_topology=save_topology,
    display_topology=display_topology,
    graph_title=graph_title,
    marker_size=marker_size,
    verbose=verbose
)
if err_msg:
    print(f"Execution aborted, {err_msg}")
    sys.exit(1)

# --------------------------------------------------------------------
# 2) Data preparation
# --------------------------------------------------------------------
err_msg = data_preparation.data_preparation()
if err_msg:
    print(f"Execution aborted, {err_msg}")
    sys.exit(1)

# --------------------------------------------------------------------
# 3) Forward-Backward Sweep + (optional) DG outer loop
# --------------------------------------------------------------------
ST.state.outer_iteration = 0
max_diff = 1.0

while max_diff > tolerance:
    # (a) 내부 반복: 전력조류 스윕
    err_msg, inner_iter = sweep_procedures.forward_backward_sweep(tolerance, max_iterations)
    ST.state.inner_iteration = inner_iter  # 디버깅용으로 확인 가능
    if err_msg:
        print(f"Execution aborted, {err_msg}")
        sys.exit(1)

    # (b) 외부 반복: DG 모드(PQV/PI) 보정
    if ST.state.has_distributed_gen:
        # 이번 라운드 DG 갱신 레지스터 초기화
        ST.state.generation_register = pd.DataFrame(columns=[
            "bus","mode","conn","kw_ph1","kvar_ph1","kw_ph2","kvar_ph2","kw_ph3","kvar_ph3","max_diff"
        ])

        # ===== PQV (전압제어 DG) =====
        if getattr(ST.state, "has_pqv_distributed_gen", False):
            wb = ST.state.working_buses.copy()
            pqv_df = getattr(ST.state, "pqv_distributed_gen", None)
            if pqv_df is not None and not pqv_df.empty:
                volt_pqv = wb[wb['id'].isin(pqv_df['bus'])][['id', 'v_ph1', 'v_ph2', 'v_ph3']].copy()
                for idx, dg in pqv_df.iterrows():
                    bus_id = dg['bus']
                    row = volt_pqv[volt_pqv['id'] == bus_id]
                    if row.empty:
                        continue

                    # 이전/현재 전압(크기)
                    old_volts = [dg['v_ph1'], dg['v_ph2'], dg['v_ph3']]
                    new_volts = [abs(row.iloc[0]['v_ph1']),
                                 abs(row.iloc[0]['v_ph2']),
                                 abs(row.iloc[0]['v_ph3'])]

                    # DG 파라미터
                    v_set = dg['kv_set'] * 1000 / (3 ** 0.5)   # 선간[kV] 가정 → 상전압[V]
                    xd = dg['xd']
                    p_phase = dg['kw_set'] * 1000 / 3          # 위상별 유효전력[W]

                    # 필요 무효 계산(Kersting 근사)
                    t = [(v_set * new_volts[i] / xd) ** 2 - p_phase ** 2 for i in range(3)]
                    t = [x if x > 0 else 0 for x in t]
                    q_ph = [ (t[i] ** 0.5) - ((new_volts[i] ** 2) / xd) for i in range(3) ]

                    qmin = dg['kvar_min'] * 1000 / 3
                    qmax = dg['kvar_max'] * 1000 / 3
                    q_ph = [min(max(q, qmin), qmax) for q in q_ph]

                    # 기존 DG 로드 제거 + 갱신된 DG(음의 부하) 추가
                    loads_df = ST.state.loads.copy()
                    mode_up = str(dg['mode']).upper()
                    ST.state.loads = loads_df[~((loads_df['bus'] == bus_id) &
                                                (loads_df['type'].str.upper() == mode_up))].copy()
                    new_load = {
                        'bus': bus_id, 'conn': dg['conn'], 'type': dg['mode'],
                        'ph_1': -(p_phase + 1j*q_ph[0]),
                        'ph_2': -(p_phase + 1j*q_ph[1]),
                        'ph_3': -(p_phase + 1j*q_ph[2]),
                    }
                    ST.state.loads = pd.concat([ST.state.loads, pd.DataFrame([new_load])], ignore_index=True)

                    # 수렴 판단 지표(상대 전압 변화)
                    diffs = []
                    for i in range(3):
                        if new_volts[i] != 0 and old_volts[i] != 0:
                            diffs.append(abs((old_volts[i] - new_volts[i]) / new_volts[i]))
                    max_volt_diff = max(diffs) if diffs else 0.0

                    # DG 테이블 업데이트(다음 라운드 대비)
                    ST.state.pqv_distributed_gen.at[idx, 'v_ph1'] = new_volts[0]
                    ST.state.pqv_distributed_gen.at[idx, 'v_ph2'] = new_volts[1]
                    ST.state.pqv_distributed_gen.at[idx, 'v_ph3'] = new_volts[2]
                    ST.state.pqv_distributed_gen.at[idx, 'w_ph1'] = p_phase
                    ST.state.pqv_distributed_gen.at[idx, 'w_ph2'] = p_phase
                    ST.state.pqv_distributed_gen.at[idx, 'w_ph3'] = p_phase
                    ST.state.pqv_distributed_gen.at[idx, 'var_ph1'] = q_ph[0]
                    ST.state.pqv_distributed_gen.at[idx, 'var_ph2'] = q_ph[1]
                    ST.state.pqv_distributed_gen.at[idx, 'var_ph3'] = q_ph[2]
                    ST.state.pqv_distributed_gen.at[idx, 'max_diff'] = max_volt_diff

                    # generation_register 기록(kW/kvar)
                    ST.state.generation_register = pd.concat([ST.state.generation_register, pd.DataFrame([{
                        'bus': bus_id, 'mode': dg['mode'], 'conn': dg['conn'],
                        'kw_ph1': p_phase/1000.0, 'kvar_ph1': q_ph[0]/1000.0,
                        'kw_ph2': p_phase/1000.0, 'kvar_ph2': q_ph[1]/1000.0,
                        'kw_ph3': p_phase/1000.0, 'kvar_ph3': q_ph[2]/1000.0,
                        'max_diff': max_volt_diff
                    }])], ignore_index=True)

        # ===== PI (전류제어 DG) =====
        if getattr(ST.state, "has_pi_distributed_gen", False):
            wb = ST.state.working_buses.copy()
            pi_df = getattr(ST.state, "pi_distributed_gen", None)
            if pi_df is not None and not pi_df.empty:
                volt_pi = wb[wb['id'].isin(pi_df['bus'])][['id', 'v_ph1', 'v_ph2', 'v_ph3']].copy()
                for idx, dg in pi_df.iterrows():
                    bus_id = dg['bus']
                    row = volt_pi[volt_pi['id'] == bus_id]
                    if row.empty:
                        continue

                    new_volts = [abs(row.iloc[0]['v_ph1']),
                                 abs(row.iloc[0]['v_ph2']),
                                 abs(row.iloc[0]['v_ph3'])]
                    old_volts = [dg.get('v_ph1', new_volts[0]),
                                 dg.get('v_ph2', new_volts[1]),
                                 dg.get('v_ph3', new_volts[2])]

                    p_phase = dg['kw_set'] * 1000 / 3
                    i_set = dg['amp_set']
                    qmin = dg['kvar_min'] * 1000 / 3
                    qmax = dg['kvar_max'] * 1000 / 3

                    conn = str(dg['conn']).upper()
                    if conn == "Y":
                        v_ph = new_volts[:]
                        i_ph = i_set
                    elif conn == "D":
                        # 델타는 선간전압 사용, 상전류는 I/√3
                        v_ph = [
                            abs(row.iloc[0]['v_ph1'] - row.iloc[0]['v_ph2']),
                            abs(row.iloc[0]['v_ph2'] - row.iloc[0]['v_ph3']),
                            abs(row.iloc[0]['v_ph3'] - row.iloc[0]['v_ph1']),
                        ]
                        i_ph = i_set / np.sqrt(3.0)
                    else:
                        v_ph = new_volts[:]
                        i_ph = i_set

                    # |S| = I*V, P 고정 → Q 산출
                    q_ph = []
                    for i in range(3):
                        val = (i_ph * v_ph[i])**2 - p_phase**2
                        qv = np.sqrt(val) if val >= 0 else qmin
                        q_ph.append(min(max(qv, qmin), qmax))

                    # 기존 DG 로드 제거 + 갱신된 DG(음의 부하) 추가
                    loads_df = ST.state.loads.copy()
                    mode_up = str(dg['mode']).upper()
                    ST.state.loads = loads_df[~((loads_df['bus'] == bus_id) &
                                                (loads_df['type'].str.upper() == mode_up))].copy()
                    new_load = {
                        'bus': bus_id, 'conn': dg['conn'], 'type': dg['mode'],
                        'ph_1': -(p_phase + 1j*q_ph[0]),
                        'ph_2': -(p_phase + 1j*q_ph[1]),
                        'ph_3': -(p_phase + 1j*q_ph[2]),
                    }
                    ST.state.loads = pd.concat([ST.state.loads, pd.DataFrame([new_load])], ignore_index=True)

                    # 수렴 판단 지표
                    diffs = []
                    for i in range(3):
                        if new_volts[i] != 0 and old_volts[i] != 0:
                            diffs.append(abs((old_volts[i] - new_volts[i]) / new_volts[i]))
                    max_volt_diff = max(diffs) if diffs else 0.0

                    # DG 테이블 업데이트
                    ST.state.pi_distributed_gen.at[idx, 'v_ph1'] = new_volts[0]
                    ST.state.pi_distributed_gen.at[idx, 'v_ph2'] = new_volts[1]
                    ST.state.pi_distributed_gen.at[idx, 'v_ph3'] = new_volts[2]
                    ST.state.pi_distributed_gen.at[idx, 'max_diff'] = max_volt_diff

                    # generation_register 기록(kW/kvar) — PI는 세 위상 동일 P 가정
                    ST.state.generation_register = pd.concat([ST.state.generation_register, pd.DataFrame([{
                        'bus': bus_id, 'mode': dg['mode'], 'conn': dg['conn'],
                        'kw_ph1': p_phase/1000.0, 'kvar_ph1': q_ph[0]/1000.0,
                        'kw_ph2': p_phase/1000.0, 'kvar_ph2': q_ph[1]/1000.0,
                        'kw_ph3': p_phase/1000.0, 'kvar_ph3': q_ph[2]/1000.0,
                        'max_diff': max_volt_diff
                    }])], ignore_index=True)

        # 이번 라운드 DG 변화량의 최대치로 외부 수렴판단
        if not ST.state.generation_register.empty and 'max_diff' in ST.state.generation_register.columns:
            max_diff = float(ST.state.generation_register['max_diff'].max())
        else:
            max_diff = 0.0
    else:
        # DG가 없으면 외부 반복은 불필요
        max_diff = 0.0

    ST.state.outer_iteration += 1

# --------------------------------------------------------------------
# 4) 결과 출력/저장
# --------------------------------------------------------------------
if not err_msg:
    if verbose != 0:
        print(f"Execution finished, {ST.state.outer_iteration} outer iterations, "
              f"{ST.state.inner_iteration} inner iterations (for latest outer round), {tolerance} tolerance")
    print_results.results(display_summary, timestamp)
else:
    print(f"Execution aborted, {err_msg}")
    sys.exit(1)
