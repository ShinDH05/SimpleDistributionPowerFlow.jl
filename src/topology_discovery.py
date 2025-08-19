from pandas import DataFrame

import shared_state as ST
import data_input
import pandas as pd
import numpy as np

def adjacency_matrix(buses, segments):
    # 1) Sort buses by ID for consistent indexing
    buses_sorted = buses.sort_values('id').reset_index(drop=True)
    n = len(buses_sorted)
    adj_mat = np.zeros((n, n), dtype=int)

    # 2) Populate adjacency for each segment
    for _, seg in segments.iterrows():
        i = buses_sorted.index[buses_sorted['id'] == seg['bus1']][0]
        j = buses_sorted.index[buses_sorted['id'] == seg['bus2']][0]
        adj_mat[i, j] = 1

    # 3) If switches exist, remove edges for open switches
    if ST.state.has_switch:
        mask = (
            ~segments['config'].isin(ST.state.line_configurations['config'])
            & segments['config'].isin(ST.state.switches['config'])
        )
        for _, seg in segments[mask].iterrows():
            i = buses_sorted.index[buses_sorted['id'] == seg['bus1']][0]
            j = buses_sorted.index[buses_sorted['id'] == seg['bus2']][0]
            state = ST.state.switches.loc[ST.state.switches['config'] == seg['config'], 'state'].iloc[0]
            if state != 'CLOSED':
                adj_mat[i, j] = 0

    return adj_mat

def gridtopology(caller: str = "",
                 input_dir: str = "",
                 output_dir: str = "",
                 save_topology: bool = True,
                 display_topology: bool = False,
                 graph_title: str = "",
                 marker_size: float = 1.5,
                 timestamp: bool = False,
                 verbose: int = 1):

    # Check for caller
    if caller not in ["user", "powerflow"]:
        print("Option caller not permitted")
        return

    # Checking Directories
    directory = input_dir
    type_f = "input_dir"
    input_dir_checked, err_msg = data_input.directory_check(directory, type_f)
    if err_msg:
        if caller == "user":
            return f"Execution aborted, {err_msg}"
        return err_msg

    directory = output_dir
    type_f = "output_dir"
    output_dir_global, err_msg = data_input.directory_check(directory, type_f)

    # Reading input files
    err_msg = data_input.read_input_files(input_dir_checked, caller, verbose)
    if err_msg:
        if caller == "user":
            print(f"Execution aborted, {err_msg}")
            return
        return err_msg

    # Identify buses from input line segments
    input_buses = pd.DataFrame(ST.state.input_segments['bus2'].unique(), columns=['id'])
    for bus in ST.state.input_segments['bus1'].unique():
        if bus not in input_buses['id'].values:
            input_buses = pd.concat([input_buses, pd.DataFrame([{'id': bus}])], ignore_index=True)

    # Defining working topology
    working_segments = ST.state.input_segments.copy()
    working_segments['check'] = 0
    if ST.state.has_switch:
        for idx, seg in working_segments.iterrows():
            for _, sw in ST.state.switches.iterrows():
                if seg['config'] == sw['config'] and sw['state'] != "CLOSED":
                    working_segments.at[idx, 'check'] = 1
        working_segments = working_segments[working_segments['check'] == 0]

    # Filtering out disconnected input segments
    # Start with existing substation buses
    working_buses = pd.DataFrame(ST.state.substation['bus'].values, columns=['id'])

    # Monitor to detect when no new bus is added
    increase_monitor = len(working_buses)

    # Iterate over working_segments in order
    for n in range(len(working_segments)):
        # Get the nth bus ID
        bus_id = working_buses.at[n, 'id']

        # Find all segments where bus1 matches and bus2 is not yet in working_buses
        new_buses = working_segments[
            (working_segments['bus1'] == bus_id) &
            (~working_segments['bus2'].isin(working_buses['id']))
            ]['bus2']

        # Append any newly discovered bus2 IDs
        for new_bus in new_buses:
            working_buses = pd.concat([working_buses, pd.DataFrame([{'id': new_bus}])], ignore_index=True)

        # If no new bus was appended, break (no further connectivity)
        if len(working_buses) > increase_monitor:
            increase_monitor += 1
        else:
            break

    # Verifying topology changes by switches
    if ST.state.has_switch and len(input_buses) != len(working_buses):
        diff = len(input_buses) - len(working_buses)
        for _ in range(diff):
            tmp = working_segments[
                working_segments['bus2'].isin(working_buses['id']) &
                ~working_segments['bus1'].isin(working_buses['id'])
            ][['bus1', 'bus2']]
            if not tmp.empty:
                bus1, bus2 = tmp.iloc[0]['bus1'], tmp.iloc[0]['bus2']
                working_buses = pd.concat([working_buses, pd.DataFrame({'id': bus1})], ignore_index=True)
                mask = (working_segments['bus1'] == bus1) & (working_segments['bus2'] == bus2)
                working_segments.loc[mask, ['bus1', 'bus2']] = working_segments.loc[mask, ['bus2', 'bus1']].values

    # Pruning out disconnected working segments
    working_segments.loc[~working_segments['bus1'].isin(working_buses['id']), 'check'] = 1
    working_segments = working_segments[working_segments['check'] == 0].drop(columns=['check'])

    # Check for loops (radial only)
    if len(working_segments) - len(working_buses) + 1 > 0:
        err_msg = (f"Topology has a loop, this version only works with radial topologies. "
                   f"See result in {output_dir_global}.")
        return err_msg

    # Adding auxiliary buses for distributed loads
    if ST.state.has_distributed_load:
        working_segments['length'] = working_segments['length'].astype(float)
        dist_load = working_segments.merge(
            ST.state.distributed_loads, on=['bus1', 'bus2'], how='inner'
        )
        # Remove original segments
        working_segments = working_segments[
            ~working_segments.set_index(['bus1', 'bus2']).index.isin(
                dist_load.set_index(['bus1', 'bus2']).index
            )
        ]
        auxiliar_buses: DataFrame = pd.DataFrame(columns=['bus1', 'bus2', 'busx'])
        next_bus_id = working_buses['id'].max() + 1
        for _, row in dist_load.iterrows():
            half_len = row['length'] / 2
            start, end = row['bus1'], row['bus2']
            unit, conf = row['unit'], row['config']
            working_segments = pd.concat(
                [working_segments,
                 pd.DataFrame([
                     {'bus1': start, 'bus2': next_bus_id, 'length': half_len, 'unit': unit, 'config': conf},
                     {'bus1': next_bus_id, 'bus2': end, 'length': half_len, 'unit': unit, 'config': conf},
                 ])],
                ignore_index=True
            )
            auxiliar_buses = pd.concat(
                [auxiliar_buses, pd.DataFrame([{'bus1': start, 'bus2': end, 'busx': next_bus_id}])],
                ignore_index=True
            )
            next_bus_id += 1
        working_buses = pd.concat([
            working_buses,
            auxiliar_buses[['busx']].rename(columns={'busx': 'id'})
        ], ignore_index=True)

    ST.state.working_buses = working_buses
    ST.state.working_segments = working_segments
    ST.state.auxiliar_buses = auxiliar_buses if 'auxiliar_buses' in locals() else None
    ST.state.adj_mat = adjacency_matrix(ST.state.working_buses, ST.state.working_segments)
    ST.state.output_dir = output_dir_global

    # Function exit by caller type
    if caller == "user":
        print(f"Execution finished, see results in {output_dir_global}.")
    else:
        return err_msg