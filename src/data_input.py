import os
import sys
import pandas as pd
import shared_state as ST
from typing import Tuple

def directory_check(dir_path: str, dir_type: str) -> Tuple[str, str]:
    err_msg = ""

    if dir_type == "input_dir":
        # 입력 타입일 때
        if dir_path != "":
            if os.path.isdir(dir_path):
                # 절대 경로가 아니면 현재 작업 디렉터리를 기준으로 절대 경로로 변환
                if not os.path.isabs(dir_path):
                    dir_path = os.path.join(os.getcwd(), dir_path)
            else:
                err_msg = f"{dir_path} is not a valid directory"
        else:
            # 빈 문자열이면 현재 작업 디렉터리 사용
            dir_path = os.getcwd()

    elif dir_type == "output_dir":
        # 출력 타입일 때
        if dir_path != "":
            if os.path.isdir(dir_path):
                if not os.path.isabs(dir_path):
                    dir_path = os.path.join(os.getcwd(), dir_path)
            else:
                # 디렉터리가 없으면 생성한 뒤 절대 경로로 변환
                os.makedirs(dir_path, exist_ok=True)
                dir_path = os.path.join(os.getcwd(), dir_path)
        else:
            dir_path = os.getcwd()

    return dir_path, err_msg

def read_file(input_directory: str, filename: str) -> Tuple[pd.DataFrame, str]:
    """
    Reads a CSV file from the given directory and returns a DataFrame plus an error message.

    Parameters:
      - input_directory: path to the folder containing the file
      - filename: name of the CSV file to read

    Returns:
      - df: pandas.DataFrame (empty if file missing)
      - err_msg: "" if successful, "empty file" if no rows, or "no file" if not found
    """
    err_msg = ""
    input_path = os.path.join(input_directory, filename)

    if os.path.isfile(input_path):
        df = pd.read_csv(input_path)
        if df.empty:
            err_msg = "empty file"
    else:
        df = pd.DataFrame()
        err_msg = "no file"

    return df, err_msg

def read_input_files(input_directory: str, caller: str, verbose: int) -> str:
    # 초기화
    ST.reset()

    has_transformer = False
    has_switch = False
    has_regulator = False
    has_bus_coords = False
    has_distributed_load = False
    has_capacitor = False
    has_distributed_gen = False

    substation = None
    input_segments = None
    line_configurations = None
    transformers = None
    switches = None
    regulators = None
    bus_coords = None
    distributed_loads = None
    spot_loads = None
    input_capacitors = None
    distributed_gen = None

    err_msg = ""
    accepted_units = ["ft", "mi", "m", "km"]

    # substation.csv
    substation, err_msg = read_file(input_directory, "substation.csv")
    if err_msg == "no file":
        return f"there is not 'substation.csv' file in {input_directory}"
    if err_msg == "empty file":
        return "'substation.csv' file is empty"
    if list(substation.columns) != ["bus", "kva", "kv"]:
        return "check for column names in 'substation.csv' file"

    # line_segments.csv
    input_segments, err_msg = read_file(input_directory, "line_segments.csv")
    if err_msg == "no file":
        return f"there is not 'line_segments.csv' file in {input_directory}"
    if err_msg == "empty file":
        return "'line_segments.csv' file is empty"
    if list(input_segments.columns) != ["bus1", "bus2", "length", "unit", "config"]:
        return "check for column names in 'line_segments.csv' file"
    if input_segments.loc[~input_segments["unit"].isin(accepted_units)].shape[0] > 0:
        return "check for units in 'line_segments.csv' file (only ft, mi, m and km are accepted units)"
    if input_segments.shape[0] != input_segments[["bus1", "bus2"]].drop_duplicates().shape[0]:
        return "check for duplicated links in 'line_segments.csv' file"
    input_segments["config"] = input_segments["config"].astype(str).str.upper()

    # line_configurations.csv
    line_configurations, err_msg = read_file(input_directory, "line_configurations.csv")
    if err_msg == "no file":
        return f"there is not 'line_configurations.csv' file in {input_directory}"
    if err_msg == "empty file":
        return "'line_configurations.csv' file is empty"
    expected_cols = [
        "config","unit","raa","xaa","rab","xab","rac","xac",
        "rbb","xbb","rbc","xbc","rcc","xcc","baa","bab",
        "bac","bbb","bbc","bcc"
    ]
    if list(line_configurations.columns) != expected_cols:
        return "check for column names in 'line_configurations.csv' file"
    if line_configurations.loc[~line_configurations["unit"].isin(accepted_units)].shape[0] > 0:
        return "check for units in 'line_configurations.csv' file (only ft, mi, m and km are accepted units)"
    if line_configurations.shape[0] != line_configurations[["config"]].drop_duplicates().shape[0]:
        return "check for duplicated configuration code in 'line_configurations.csv' file"
    line_configurations["config"] = line_configurations["config"].astype(str).str.upper()

    # transformers.csv
    transformers, err_msg = read_file(input_directory, "transformers.csv")
    if err_msg == "":
        if list(transformers.columns) != ["config", "kva", "phases", "conn_high", "conn_low", "kv_high", "kv_low", "rpu", "xpu"]:
            return "check for column names in 'transformers.csv' file"
        transformers["config"] = transformers["config"].astype(str).str.upper()
        if transformers.shape[0] != transformers[["config"]].drop_duplicates().shape[0]:
            return "check for duplicated configuration code in 'transformers.csv' file"
        transformers["conn_high"] = transformers["conn_high"].str.upper()
        transformers["conn_low"] = transformers["conn_low"].str.upper()
        has_transformer = True

    # switches.csv
    switches, err_msg = read_file(input_directory, "switches.csv")
    if err_msg == "":
        if list(switches.columns) != ["config", "phases", "state", "resistance"]:
            return "check for column names in 'switches.csv' file"
        switches["config"] = switches["config"].astype(str).str.upper()
        if switches.shape[0] != switches[["config"]].drop_duplicates().shape[0]:
            return "check for duplicated configuration code in 'switches.csv' file"
        switches["state"] = switches["state"].str.upper()
        has_switch = True

    # regulators.csv
    regulators, err_msg = read_file(input_directory, "regulators.csv")
    if err_msg == "":
        if list(regulators.columns) != ["config", "phases", "mode", "tap_1", "tap_2", "tap_3"]:
            return "check for column names in 'regulators.csv' file."
        regulators["config"] = regulators["config"].astype(str).str.upper()
        if regulators.shape[0] != regulators[["config"]].drop_duplicates().shape[0]:
            return "check for duplicated configuration code in 'regulators.csv' file"
        regulators["mode"] = regulators["mode"].str.upper()
        has_regulator = True

    # bus_coords.csv
    bus_coords, err_msg = read_file(input_directory, "bus_coords.csv")
    if err_msg == "":
        if list(bus_coords.columns) != ["bus", "x", "y"]:
            print("check for column names in 'bus_coords.csv' file. Following without bus coordinates.")
        elif bus_coords.shape[0] != bus_coords[["bus"]].drop_duplicates().shape[0]:
            print("check for duplicated bus name in 'bus_coords.csv' file. Following without bus coordinates.")
        else:
            has_bus_coords = True

    #Identifying input segments types other than line type
    without_config = input_segments.loc[~input_segments["config"].isin(line_configurations["config"])]
    if not without_config.empty and has_transformer:
        without_config = without_config.loc[~without_config["config"].isin(transformers["config"])]
    if not without_config.empty and has_switch:
        without_config = without_config.loc[~without_config["config"].isin(switches["config"])]
    if not without_config.empty and has_regulator:
        without_config = without_config.loc[~without_config["config"].isin(regulators["config"])]
    if not without_config.empty:
        return f"check for {without_config['config'].tolist()} code(s) in 'line_segments', 'line_configurations', 'transformers', 'switches' or 'regulators' .csv files in {input_directory}"

    # distributed_loads.csv
    distributed_loads, err_msg = read_file(input_directory, "distributed_loads.csv")
    if err_msg == "":
        if list(distributed_loads.columns) != ["bus1","bus2","conn","type","kw_ph1","kvar_ph1","kw_ph2","kvar_ph2","kw_ph3","kvar_ph3"]:
            return "check for column names in 'distributed_loads.csv' file"
        has_distributed_load = True
    else:
        if verbose != 0:
            print("no distributed loads")
        err_msg = ""

    if caller == "powerflow":
        # spot_loads.csv
        spot_loads, err_msg = read_file(input_directory, "spot_loads.csv")
        if err_msg == "no file":
            return f"there is not 'spot_loads.csv' file in {input_directory}"
        if err_msg == "empty file":
            return "'spot_loads.csv' file is empty"
        if list(spot_loads.columns) != ["bus","conn","type","kw_ph1","kvar_ph1","kw_ph2","kvar_ph2","kw_ph3","kvar_ph3"]:
            return "check for column names in 'spot_loads.csv' file"

        # capacitors.csv
        input_capacitors, err_msg = read_file(input_directory, "capacitors.csv")
        if err_msg == "":
            if list(input_capacitors.columns) != ["bus","kvar_ph1","kvar_ph2","kvar_ph3"]:
                return "check for column names in 'capacitors.csv' file"
            has_capacitor = True
        else:
            if verbose != 0:
                print("no capacitors")
            err_msg = ""

        # distributed_generation.csv
        distributed_gen, err_msg = read_file(input_directory, "distributed_generation.csv")
        if err_msg == "":
            if list(distributed_gen.columns) != ["bus","conn","mode","kw_set","kvar_set","kv_set","amp_set","kvar_min","kvar_max","xd"]:
                return "check for column names in 'distributed_generation.csv' file"
            distributed_gen["mode"] = distributed_gen["mode"].str.upper()
            invalid_modes = distributed_gen.loc[~distributed_gen["mode"].isin(["PQ","PQV","PI"])]
            if not invalid_modes.empty:
                return "modes of Distributed Generation accepted: PQ (traditional constant watt-var), PQV (volt dependant var PQ), PI (constant watt-ampere)"
            if distributed_gen.dropna(subset=["bus","conn","mode"]).shape[0] != distributed_gen.shape[0]:
                print("Distributed generation registers with missing values will be ignored.")
                distributed_gen = distributed_gen.dropna(subset=["bus","conn","mode"])
            has_distributed_gen = not distributed_gen.empty
        else:
            if verbose != 0:
                print("no distributed generation")
            err_msg = ""

    ST.state.substation          = substation
    ST.state.input_segments      = input_segments
    ST.state.line_configurations = line_configurations
    ST.state.transformers        = transformers
    ST.state.switches            = switches
    ST.state.regulators          = regulators
    ST.state.bus_coords          = bus_coords
    ST.state.distributed_loads   = distributed_loads
    ST.state.spot_loads          = spot_loads
    ST.state.input_capacitors    = input_capacitors
    ST.state.distributed_gen     = distributed_gen

    ST.state.has_transformer      = has_transformer
    ST.state.has_switch           = has_switch
    ST.state.has_regulator        = has_regulator
    ST.state.has_bus_coords       = has_bus_coords
    ST.state.has_distributed_load = has_distributed_load
    ST.state.has_capacitor        = has_capacitor
    ST.state.has_distributed_gen  = has_distributed_gen

    return err_msg

def main():
    input_dir = r"C:\dev\ACPF\BFS\examples\ieee-13"
    output_dir = r"C:\dev\ACPF\BFS\results"
    caller = "powerflow"
    verbose = 1

    # validate input_dir
    inp_checked, err = directory_check(input_dir, "input_dir")
    if err:
        print(f"Execution aborted: {err}")
        sys.exit(1)

    # validate output_dir
    out_checked, err = directory_check(output_dir, "output_dir")
    if err:
        print(f"Execution aborted: {err}")
        sys.exit(1)

    # read input files
    err = read_input_files(inp_checked, caller, verbose)
    if err:
        print(f"Execution aborted: {err}")
        sys.exit(1)

    print("All files loaded successfully.")

if __name__ == "__main__":
    main()
