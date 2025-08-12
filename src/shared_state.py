from dataclasses import dataclass, field
import pandas as pd
import numpy as np

@dataclass
class State:
    # Input data tables (from CSV files):contentReference[oaicite:0]{index=0}
    substation: pd.DataFrame = field(default_factory=pd.DataFrame)  # substation.csv (bus, kva, kv)
    input_segments: pd.DataFrame = field(default_factory=pd.DataFrame)  # line_segments.csv (bus1, bus2, length, unit, config)
    line_configurations: pd.DataFrame = field(default_factory=pd.DataFrame)  # line_configurations.csv (config, unit, raa, xaa, ..., bcc)
    transformers: pd.DataFrame = field(default_factory=pd.DataFrame)  # transformers.csv (config, kva, phases, conn_high, conn_low, kv_high, kv_low, rpu, xpu)
    switches: pd.DataFrame = field(default_factory=pd.DataFrame)  # switches.csv (config, phases, state, resistance)
    regulators: pd.DataFrame = field(default_factory=pd.DataFrame)  # regulators.csv (config, phases, mode, tap_1, tap_2, tap_3)
    distributed_loads: pd.DataFrame = field(default_factory=pd.DataFrame)  # distributed_loads.csv (bus1, bus2, conn, type, kw_ph1, kvar_ph1, ...)
    spot_loads: pd.DataFrame = field(default_factory=pd.DataFrame)  # spot_loads.csv (bus, conn, type, kw_ph1, kvar_ph1, ...)
    input_capacitors: pd.DataFrame = field(default_factory=pd.DataFrame)  # capacitors.csv (bus, kvar_ph1, kvar_ph2, kvar_ph3)
    distributed_gen: pd.DataFrame = field(default_factory=pd.DataFrame)  # distributed_generation.csv (bus, conn, mode, kw_set, kvar_set, kv_set, amp_set, kvar_min, kvar_max, xd)
    bus_coords: pd.DataFrame = field(default_factory=pd.DataFrame)  # bus_coords.csv (bus, x, y)

    # Flags for presence of optional data:contentReference[oaicite:1]{index=1}
    has_transformer: bool = False
    has_switch: bool = False
    has_regulator: bool = False
    has_distributed_load: bool = False
    has_capacitor: bool = False
    has_distributed_gen: bool = False
    has_bus_coords: bool = False

    # Network topology data structures:contentReference[oaicite:2]{index=2}
    working_segments: pd.DataFrame = field(default_factory=pd.DataFrame)  # adjusted line segments (after removing open switches, etc.)
    working_buses: pd.DataFrame = field(default_factory=pd.DataFrame)  # all buses in working topology (with types and numbers)
    auxiliar_buses: pd.DataFrame = field(default_factory=pd.DataFrame)  # dummy/auxiliary buses added (e.g., for distributed loads)
    auxiliar_segments: pd.DataFrame = field(default_factory=pd.DataFrame)  # dummy line segments connecting auxiliary buses
    adj_mat: np.ndarray = None  # adjacency matrix for bus connectivity (NumPy 2D array):contentReference[oaicite:3]{index=3}

    # Base voltage values and transformation matrices:contentReference[oaicite:4]{index=4}:contentReference[oaicite:5]{index=5}
    ELN: np.ndarray = None  # nominal line-to-neutral voltages (complex, length 3):contentReference[oaicite:6]{index=6}
    ELL: np.ndarray = None  # nominal line-to-line voltages (complex, length 3):contentReference[oaicite:7]{index=7}
    D: np.ndarray = field(default_factory=lambda: np.array([[1, -1, 0], [0, 1, -1], [-1, 0, 1]], dtype=float))  # matrix to convert phase-to-line voltages:contentReference[oaicite:8]{index=8}
    a_exp: complex = None  # complex 120° rotation factor (Julia `as`):contentReference[oaicite:9]{index=9}
    As: np.ndarray = None  # 3x3 transformation matrix using a_exp (for sequence components):contentReference[oaicite:10]{index=10}

    # Processed line and configuration data for calculations
    line_configs: pd.DataFrame = field(default_factory=pd.DataFrame)  # working line configurations with complex impedances:contentReference[oaicite:11]{index=11}
    lines: pd.DataFrame = field(default_factory=pd.DataFrame)  # working lines (all line/xfmr/switch segments with impedance & phase data):contentReference[oaicite:12]{index=12}

    # Generalized line impedance matrices (for forward/backward sweep):contentReference[oaicite:13]{index=13}
    gen_lines_mat: pd.DataFrame = field(default_factory=pd.DataFrame)  # per-line matrices (a, b, c, d, A, B components) for sweeps

    # Aggregated load data and distributed generation registers
    loads: pd.DataFrame = field(default_factory=pd.DataFrame)  # combined loads table (spot, distributed, capacitors, gen as negative):contentReference[oaicite:14]{index=14}
    pq_distributed_gen: pd.DataFrame = field(default_factory=pd.DataFrame)  # subset of distributed_gen for PQ mode:contentReference[oaicite:15]{index=15}
    pqv_distributed_gen: pd.DataFrame = field(default_factory=pd.DataFrame)  # subset of distributed_gen for PQV mode (with voltage setpoint):contentReference[oaicite:16]{index=16}
    pi_distributed_gen: pd.DataFrame = field(default_factory=pd.DataFrame)  # subset of distributed_gen for PI mode (constant current):contentReference[oaicite:17]{index=17}
    generation_register: pd.DataFrame = field(default_factory=pd.DataFrame)  # tracking generation output (P/Q per phase, etc.):contentReference[oaicite:18]{index=18}:contentReference[oaicite:19]{index=19}

    # Temporary vectors for voltages and currents (3x1 complex):contentReference[oaicite:20]{index=20}:contentReference[oaicite:21]{index=21}
    Vbus1: np.ndarray = field(default_factory=lambda: np.zeros((3, 1),
                                                               dtype=complex))  # backward-sweep upstream bus voltage (placeholder):contentReference[oaicite:22]{index=22}
    Vbus2: np.ndarray = field(default_factory=lambda: np.zeros((3, 1),
                                                               dtype=complex))  # forward-sweep downstream bus voltage (placeholder):contentReference[oaicite:23]{index=23}
    Ibus1: np.ndarray = field(default_factory=lambda: np.zeros((3, 1),
                                                               dtype=complex))  # backward-sweep upstream bus current (placeholder):contentReference[oaicite:24]{index=24}
    Ibus2: np.ndarray = field(default_factory=lambda: np.zeros((3, 1),
                                                               dtype=complex))  # forward-sweep downstream bus current (placeholder):contentReference[oaicite:25]{index=25}
    Iline: np.ndarray = field(default_factory=lambda: np.zeros((3, 1),
                                                               dtype=complex))  # line currents in sequence for delta loads:contentReference[oaicite:26]{index=26}
    Iphase: np.ndarray = field(default_factory=lambda: np.zeros((3, 1),
                                                                dtype=complex))  # phase currents for delta-connected load calc:contentReference[oaicite:27]{index=27}

    # Symmetrical sequence data (if needed)
    x_seq_df: pd.DataFrame = field(default_factory=pd.DataFrame)  # DataFrame for sequence components (if used):contentReference[oaicite:28]{index=28}
    x_seq: np.ndarray = None  # array or vector for sequence values (if used):contentReference[oaicite:29]{index=29}

    # Iteration counters for solver loops
    outer_iteration: int = 0  # outer loop counter in powerflow (overall iterations):contentReference[oaicite:30]{index=30}
    inner_iteration: int = 0  # inner loop counter or last iteration count in sweep:contentReference[oaicite:31]{index=31}

    # Dictionary route
    output_dir: str = ""

state = State()

def reset():
    """테스트/재실행 시 상태 초기화"""
    global state
    state = State()
