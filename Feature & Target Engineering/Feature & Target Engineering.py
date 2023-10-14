# Required Libraries
import pandas as pd
import numpy as np
import gzip
import os
import matplotlib.pyplot as plt 
import seaborn as sns
import json

# Turn a string into a list of dictionaries to access individual entries
def parse_json(entry):
    valid_json = entry.replace("'", "\"")
    return json.loads(valid_json)

""" Dataset Engineering Process """

# Group all recorded liquidity events in a DataFrame
def extract_liquidity_events(folder_name):
    liquidity_events = pd.DataFrame()
    
    # Get a list of all files within the liquidity events folder
    file_list = os.listdir(folder_name)
    
    # Iterate over all but the last file within the "full_liquidity_events" folder
    # The June 29th file is incomplete, so avoid considering it
    for idx, file in enumerate(file_list[:-1]):
        with gzip.open("full_liquidity_events/" + file) as f:
            daily_events = pd.read_csv(f)
        liquidity_events = pd.concat([liquidity_events, daily_events], axis=0)
    

    # Manipulate column data types for future operations, plots, and tables  
    liquidity_events['type'] = liquidity_events['type'].astype('category')
    liquidity_events['exchange'] = liquidity_events['exchange'].astype('category')
    liquidity_events['pool_name'] = liquidity_events['pool_name'].astype('category')
    liquidity_events.rename(columns={'Unnamed: 0': 'date'}, inplace=True)
    liquidity_events['date'] = pd.to_datetime(liquidity_events['date'], format="%Y-%m-%d %H:%M:%S")
    
    # Sort events by date and reset index
    liquidity_events.sort_values(by='date', inplace=True)
    liquidity_events.reset_index(drop=True, inplace=True)
    
    # Drop columns we don't use
    liquidity_events.drop(columns=['transaction_hash', 'log_index'], inplace=True)
    
    return liquidity_events

# Liquidity events DataFrame
full_events = extract_liquidity_events("full_liquidity_events")

# Relevant folder names
usdc_weth_5_folder = "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640"
usdc_weth_30_folder = "0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8"


""" USDC-WETH-0.0005 Dataset """

""" Contemporaneous Liquidity Features """

# Create liquidity events DataFrames for pools USDC-WETH-0.0005 and USDC-WETH-0.003
usdc_weth_5_events = full_events[(full_events["pool_name"]=="USDC-WETH-0.0005")].copy()
usdc_weth_5_events.reset_index(drop=True, inplace=True)
usdc_weth_30_events = full_events[(full_events["pool_name"]=="USDC-WETH-0.003")].copy()
usdc_weth_30_events.reset_index(drop=True, inplace=True)

# Helper function for creating the liquidity_type feature 
def determine_liquidity_type(row):
    if not np.isnan(row['s0_usdc']) and not np.isnan(row['s0_weth']):
        return "active"
    else:
        return "inactive"

# Helper function to calculate the USD size of a liquidity event
def usd_volume(row):
    usdc = row['s0_usdc'] if not pd.isnull(row['s0_usdc']) else 0
    weth = row['s0_weth'] if not pd.isnull(row['s0_weth']) else 0
    return (usdc * 1) + (weth / row['price'])

# Adds contemporaneous liquidity features to the input DataFrame
def contemporaneous_liquidity_features(df):
    df_contemp = df

    # Get USDC and WETH liquidity event amounts 
    df_contemp['amounts'] = df_contemp['amounts'].apply(parse_json)
    df_contemp['s0_usdc'] = df_contemp['amounts'].apply(lambda x: x[0].get('amount') if x else None)
    df_contemp['s0_weth'] = df_contemp['amounts'].apply(lambda x: x[1].get('amount') if x else None)
    
    # Drop all non-events (as recommended by Kaiko) and reset index
    df_contemp.dropna(subset=['s0_usdc', 's0_weth'], how = 'all', inplace = True)
    df_contemp.reset_index(drop=True, inplace=True)
    
    # Get the USD size of liquidity events
    df_contemp['s0'] = df_contemp.apply(usd_volume, axis = 1)
    
    # Get the tick range of liquidity events
    df_contemp['ticker_range'] = df_contemp['metadata'].apply(parse_json)
    df_contemp['w0'] = df_contemp['ticker_range'].apply(lambda x: x['upper_ticker'] - x['lower_ticker'])
    
    # Create the liquidity_type feature
    df_contemp['liquidity_type'] = df_contemp.apply(determine_liquidity_type, axis=1).astype('category')
    
    # Drop unused columns
    df_contemp.drop(columns=['blockchain', 'pool_address', 'exchange', 'amounts',
                                     'metadata', 'user_address', 'ticker_range', 
                                     's0_usdc', 's0_weth'], inplace=True)
    
    return df_contemp

# Add contemporaneous liquidity features to the Pool 5 and Pool 30 DataFrames
usdc_weth_5_contemp = contemporaneous_liquidity_features(usdc_weth_5_events)
usdc_weth_30_contemp = contemporaneous_liquidity_features(usdc_weth_30_events)


""" Block-Wise Aggregated Events """

# Aggregates all liquidity events on the same block as in Algorithm 2
def blockwise_events(group):
    # Sum USD sizes of both types of event
    mint_sum = group.loc[group['type'] == 'mint', 's0'].sum()
    burn_sum = group.loc[group['type'] == 'burn', 's0'].sum()
    
    # Classify as mint or burn based on which sum is higher
    if mint_sum > burn_sum:
        group['type'] = 'mint'
        # Calculate the weighted average of w0 using mint type s0
        mint_s0_sum = group.loc[group['type'] == 'mint', 's0'].sum()
        group['w0'] = (group['w0'] * group['s0']).sum() / mint_s0_sum
        # Calculate the weighted majority liquidity_type using mint type s0
        mint_group = group.loc[group['type'] == 'mint']
        liquidity_mode = mint_group.groupby('liquidity_type')['s0'].sum().idxmax()
        group['liquidity_type'] = liquidity_mode
    else:
        group['type'] = 'burn'
        # Calculate the weighted average of w0 using burn type s0
        burn_s0_sum = group.loc[group['type'] == 'burn', 's0'].sum()
        group['w0'] = (group['w0'] * group['s0']).sum() / burn_s0_sum
        # Calculate the weighted majority liquidity_type using burn type s0
        burn_group = group.loc[group['type'] == 'burn']
        liquidity_mode = burn_group.groupby('liquidity_type')['s0'].sum().idxmax()
        group['liquidity_type'] = liquidity_mode

    # Calculate the block's USD size
    group['s0'] = abs(mint_sum - burn_sum)
    
    return group

# Aggregate liquidity events block-wise and keep events of USD size > 100 for Pool 5
usdc_weth_5_block = usdc_weth_5_contemp.groupby('block_number', group_keys=False).apply(blockwise_events)
usdc_weth_5_block.drop_duplicates(keep='first', inplace=True)
usdc_weth_5_block = usdc_weth_5_block[usdc_weth_5_block["s0"] >= 100]
usdc_weth_5_block.reset_index(drop=True, inplace=True)

# Aggregate liquidity events block-wise and keep events of USD size > 100 for Pool 30
usdc_weth_30_block = usdc_weth_30_contemp.groupby('block_number', group_keys=False).apply(blockwise_events)
usdc_weth_30_block.drop_duplicates(keep='first', inplace=True)
usdc_weth_30_block = usdc_weth_30_block[usdc_weth_30_block["s0"] >= 100]
usdc_weth_30_block.reset_index(drop=True, inplace=True)


""" Lagged Main-Pool Features """

# Adds Lagged Main-Pool features to the input DataFrame
def lagged_main_features(df, n=1, event_type='mint'):
    df_lagged_main = df
    
    last_n_block_numbers = []
    last_n_widths = []
    last_n_sizes = []
    
    # Event distance lags 
    def calculate_blocks_since(row):
        nonlocal last_n_block_numbers
        if row['type'] == event_type:
            # Create NaN values if there are not enough events to take the n-th lag
            if len(last_n_block_numbers) < n:
                last_n_block_numbers.append(row['block_number'])
                return None
            else:
                blocks_since = row['block_number'] - last_n_block_numbers[0]
                last_n_block_numbers.append(row['block_number'])
                last_n_block_numbers.pop(0)
                return blocks_since
        else:
            # Create NaN values if there are not enough events to take the n-th lag
            if len(last_n_block_numbers) < n:
                return None
            else:
                return row['block_number'] - last_n_block_numbers[0]

    history_width = []
    history_size = []

    for _, row in df_lagged_main.iterrows():
        row_type, row_w0, row_s0 = row['type'], row['w0'], row['s0']
        
        # Create NaN values if there are not enough events to take the n-th lag
        if len(last_n_widths) >= n:
            history_width.append(last_n_widths[-n])
        else:
            history_width.append(None)
        # Create NaN values if there are not enough events to take the n-th lag
        if len(last_n_sizes) >= n:
            history_size.append(last_n_sizes[-n])
        else:
            history_size.append(None)
            
        # Add the current event to the list of events if appropriate 
        if row_type == event_type:
            last_n_widths.append(row_w0)
            last_n_sizes.append(row_s0)
            
    # Create 3 columns with the specified n-th lag of the chosen event_type
    df_lagged_main[f'b{n}_{event_type}_main'] = df_lagged_main.apply(calculate_blocks_since, axis=1)
    df_lagged_main[f'w{n}_{event_type}_main'] = history_width
    df_lagged_main[f's{n}_{event_type}_main'] = history_size

    return df_lagged_main

# Add Lagged Main-Pool Features to the Pool 5 DataFrame
lags = [1, 2, 3]
event_types = ["mint", "burn"]
for e_type in event_types:
    for lag in lags:
        usdc_weth_5_block = lagged_main_features(usdc_weth_5_block, n = lag, event_type = e_type)


""" Lagged Other-Pool Features """

# Adds Lagged Other-Pool features to the input DataFrame
def lagged_other_features(df, n=1, event_type='mint', other_pool='USDC-WETH-0.003'):
    df_lagged_other = df
    
    last_n_block_numbers_other = []
    last_n_widths_other = []
    last_n_sizes_other = []

    # Event distance lags
    def calculate_blocks_since_other(row):
        nonlocal last_n_block_numbers_other
        
        # Only track lags for events in the other pool
        if row['pool_name'] == other_pool:
            if row['type'] == event_type:
                if len(last_n_block_numbers_other) < n:
                    last_n_block_numbers_other.append(row['block_number'])
                    return None
                else:
                    last_n_block_numbers_other.append(row['block_number'])
                    last_n_block_numbers_other.pop(0)
                    return None
        # Only rows which are not in the other pool need non-NaN values
        else:
            # Create NaN values if there are not enough events to take the n-th lag
            if len(last_n_block_numbers_other) < n:
                return None
            else:
                return row['block_number'] - last_n_block_numbers_other[0]
                    
        return None

    history_width_other = []
    history_size_other = []

    for _, row in df_lagged_other.iterrows():
        row_type, row_w0, row_s0, row_pool_name = row['type'], row['w0'], row['s0'], row['pool_name']
        
        # Only track lags for events in the other pool
        if row_pool_name == other_pool:
            # Rows satisfying this condition will be removed anyway
            history_width_other.append(None)
            history_size_other.append(None)
            
            # Add the current event to the list of events if event_type is appropriate
            if row_type == event_type:
                last_n_widths_other.append(row_w0)
                last_n_sizes_other.append(row_s0)
        
        else:
            # Create NaN values if there are not enough events to take the n-th lag
            if len(last_n_widths_other) >= n:
                history_width_other.append(last_n_widths_other[-n])
            else:
                history_width_other.append(None)
            # Create NaN values if there are not enough events to take the n-th lag
            if len(last_n_sizes_other) >= n:
                history_size_other.append(last_n_sizes_other[-n])
            else:
                history_size_other.append(None) 
            
    # Create 3 columns with the specified n-th lag of the chosen event_type
    df_lagged_other[f'b{n}_{event_type}_other'] = df_lagged_other.apply(calculate_blocks_since_other, axis=1)
    df_lagged_other[f'w{n}_{event_type}_other'] = history_width_other
    df_lagged_other[f's{n}_{event_type}_other'] = history_size_other

    return df_lagged_other

# Concatenate events from Pool 30 to the Pool 5 DataFrame and reset index
# Place events in Pool 30 first to break block_number ties so they are not considered future events
usdc_weth_5_concat = pd.concat([usdc_weth_5_block, usdc_weth_30_block]).sort_values(by=['block_number', 'pool_name'], ascending=[True, False])
usdc_weth_5_concat.reset_index(drop=True, inplace=True)

# Add Lagged Other-Pool Features to the Pool 5 DataFrame
lags = [1, 2, 3]
event_types = ["mint", "burn"]
for e_type in event_types:
    for lag in lags:
        usdc_weth_5_concat = lagged_other_features(usdc_weth_5_concat, n = lag, event_type = e_type)


""" Classification & Regression Targets """

# Add 2 classification targets to the input DataFrame
def classification_targets(df, main_pool, other_pool):
    df_class = df

    # Initialize the empty targets
    df_class['next_type_main'] = None
    df_class['next_type_other'] = None

    for index, row in df_class.iterrows():
        # Get the index of the next event in each pool
        main_pool_idx = df_class[(df_class['pool_name'] == main_pool) & (df_class['block_number'] > row['block_number'])].index.min()
        other_pool_idx = df_class[(df_class['pool_name'] == other_pool) & (df_class['block_number'] > row['block_number'])].index.min()
        
        # Fill targets with the respective type of the next event
        if not pd.isnull(main_pool_idx):
            df_class.at[index, 'next_type_main'] = df_class.at[main_pool_idx, 'type']
    
        if not pd.isnull(other_pool_idx):
            df_class.at[index, 'next_type_other'] = df_class.at[other_pool_idx, 'type']
    
    return df_class

# Add 4 regression targets to the input DataFrame
def regression_targets(df, main_pool, other_pool):
    df_reg = df

    #Initialize the empty targets
    df_reg['next_mint_time_main'] = None
    df_reg['next_burn_time_main'] = None
    df_reg['next_mint_time_other'] = None
    df_reg['next_burn_time_other'] = None
    
    for index, row in df_reg.iterrows():
        # Get the index of the next event of each type on each pool
        next_main_mint_idx = df_reg[(df_reg['pool_name'] == main_pool) & (df_reg['type'] == 'mint') & (df_reg['block_number'] > row['block_number'])].index.min()
        next_main_burn_idx = df_reg[(df_reg['pool_name'] == main_pool) & (df_reg['type'] == 'burn') & (df_reg['block_number'] > row['block_number'])].index.min()
        next_other_mint_idx = df_reg[(df_reg['pool_name'] == other_pool) & (df_reg['type'] == 'mint') & (df_reg['block_number'] > row['block_number'])].index.min()
        next_other_burn_idx = df_reg[(df_reg['pool_name'] == other_pool) & (df_reg['type'] == 'burn') & (df_reg['block_number'] > row['block_number'])].index.min()
        
        # Fill targets with the respective block distances for each case 
        if not pd.isnull(next_main_mint_idx):
            df_reg.at[index, 'next_mint_time_main'] = df_reg.at[next_main_mint_idx, 'block_number'] - row['block_number']

        if not pd.isnull(next_main_burn_idx):
            df_reg.at[index, 'next_burn_time_main'] = df_reg.at[next_main_burn_idx, 'block_number'] - row['block_number']

        if not pd.isnull(next_other_mint_idx):
            df_reg.at[index, 'next_mint_time_other'] = df_reg.at[next_other_mint_idx, 'block_number'] - row['block_number']

        if not pd.isnull(next_other_burn_idx):
            df_reg.at[index, 'next_burn_time_other'] = df_reg.at[next_other_burn_idx, 'block_number'] - row['block_number']
    
    return df_reg

# Add 2 classification targets to the Pool 5 DataFrame
usdc_weth_5_class = classification_targets(usdc_weth_5_concat, main_pool = 'USDC-WETH-0.0005', other_pool = 'USDC-WETH-0.003')

# Add 4 regression targets to the Pool 5 DataFrame
usdc_weth_5_targets = regression_targets(usdc_weth_5_class,  main_pool = 'USDC-WETH-0.0005', other_pool = 'USDC-WETH-0.003')


""" Price & Market Depth Features """

# Remove Pool 30 liquidity events from the Pool 5 DataFrame and reset index
usdc_weth_5_targets = usdc_weth_5_targets[usdc_weth_5_targets['pool_name'] == 'USDC-WETH-0.0005']
usdc_weth_5_targets.reset_index(drop=True, inplace=True)

# Helper function for engineering market depth for a snapshot
def extract_market_depth(snapshots, current_tick):
    active_liquidity = None
    for d in snapshots:
        if d['amount0'] > 0 and d['amount1'] > 0:
            active_liquidity = d
    
    if active_liquidity is not None:
        return active_liquidity

    # Create a dictionary for a snapshot where the price is on a tick spacing endpoint
    lower_tick_entry = next((entry for entry in snapshots if entry.get('lower_tick') == current_tick), None)
    upper_tick_entry = next((entry for entry in snapshots if entry.get('upper_tick') == current_tick), None)

    if lower_tick_entry and upper_tick_entry:
        return {
            'amount0': lower_tick_entry['amount0'],
            'amount1': upper_tick_entry['amount1'],
            'amount': upper_tick_entry['amount'],  # 'amount' doesn't matter
            'lower_tick': current_tick,
            'upper_tick': current_tick
        }

    return None

# Helper function to calculate the USD size of market depth
def calculate_usd_snapshot(row):
    amount_dict = row['snapshot']
    return (amount_dict['amount0']*1) + (amount_dict['amount1'] / row['current_price'])


# Create a DataFrame of snapshots for a USP3 pool with engineered market depth
def engineer_snapshots(folder_name, price_name, snapshot_name):
    snapshots_df = pd.DataFrame()
    
    # Iterate over all files in the input folder to get a pool's snapshots
    for file in os.listdir(f'usp3_data/{folder_name}'):
        with gzip.open(f'usp3_data/{folder_name}/' + file) as f:
            daily_snapshots = pd.read_csv(f)
        snapshots_df = pd.concat([snapshots_df, daily_snapshots], axis=0)
    
    # Sort by block_number and reset index
    snapshots_df.sort_values(by="block_number", inplace=True)
    snapshots_df.reset_index(drop=True, inplace=True)
    
    # Column adjustments
    snapshots_df['pool_name'] = snapshots_df['pool_name'].astype('category')
    snapshots_df.rename(columns={'Unnamed: 0': 'date'}, inplace=True)
    snapshots_df['date'] = pd.to_datetime(snapshots_df['date'], format="%Y-%m-%d %H:%M:%S")
    
    
    # Engineer market depth feature for each snapshot
    snapshots_df['snapshots'] = snapshots_df['snapshots'].apply(parse_json)
    snapshots_df['snapshot'] = snapshots_df.apply(lambda row: 
                                                  extract_market_depth(row['snapshots'], row['current_tick']), axis=1)
    
    
    # Get USD size of market depth
    snapshots_df['usd_snapshot'] = snapshots_df.apply(calculate_usd_snapshot, axis = 1)
    
    # Drop columns we don't use
    snapshots_df.drop(columns=['pool_address', 'blockchain', 'exchange'], inplace=True)
    
    # Rename feature columns
    snapshots_df.rename(columns={'current_price': price_name, 'usd_snapshot': snapshot_name}, inplace=True)
    
    return snapshots_df

# Relevant folder names 
usdc_weth_5_folder = "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640"
usdc_weth_30_folder = "0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8"

# Create snapshot DataFrames for pools USDC-WETH-0.0005 and USDC-WETH-0.003
usdc_weth_5_snapshots = engineer_snapshots(usdc_weth_5_folder, "price_5", "snapshot_5")
usdc_weth_30_snapshots = engineer_snapshots(usdc_weth_30_folder, "price_30", "snapshot_30")

# Adds non-event snapshot observations to the input DataFrame
def non_event_blocks(df):
    non_event_blocks = []
    
    for i in range(df['block_number'].min(), df['block_number'].max() + 1):
        if i not in df['block_number'].values:
            non_event_blocks.append({'block_number': i})
    
    df_full = pd.concat([df, pd.DataFrame(non_event_blocks)], axis = 0)
    df_full.sort_values(by='block_number', inplace=True)
    df_full.reset_index(drop=True, inplace=True)
    
    df_full['type'] = df_full['type'].fillna('snap')
    
    return df_full

# Add snapshot (non-event) observations to the Pool 5 DataFrame
usdc_weth_5_full = non_event_blocks(usdc_weth_5_targets)

# Forward fills the missing snapshot information from the external data provider
def kaiko_missing(df, price_name, snapshot_name):
    missing_blocks = []
    
    for i in range(df['block_number'].min(), df['block_number'].max() + 1):
        if i not in df['block_number'].values:
            missing_blocks.append({'block_number': i})
    
    df_full_snapshots = pd.concat([df, pd.DataFrame(missing_blocks)])
    
    df_full_snapshots.sort_values(by="block_number", inplace=True)
    df_full_snapshots.reset_index(drop=True, inplace=True)
    
    df_full_snapshots[[price_name, snapshot_name]] = df_full_snapshots[[price_name, snapshot_name]].fillna(method="ffill")
    
    return df_full_snapshots

# Update snapshot DataFrames with the forward filling strategy for missing snapshots
usdc_weth_5_full_snapshots = kaiko_missing(usdc_weth_5_snapshots, "price_5", "snapshot_5")
usdc_weth_30_full_snapshots = kaiko_missing(usdc_weth_30_snapshots, "price_30", "snapshot_30")

# Adds block-wise price and market depth columns from both pools to the input DataFrame
def merged_snapshots(events_df):
    columns_to_merge_5 = ['block_number', 'price_5', 'snapshot_5']
    columns_to_merge_30 = ['block_number', 'price_30', 'snapshot_30']
    
    merged_df_int = events_df.merge(usdc_weth_5_full_snapshots[columns_to_merge_5], on='block_number', how='left')
    
    merged_df_full = merged_df_int.merge(usdc_weth_30_full_snapshots[columns_to_merge_30], on='block_number', how='left')
    
    return merged_df_full

# Add price and market depth columns from both pools to the Pool 5 DataFrame
usdc_weth_5_events_snapshots = merged_snapshots(usdc_weth_5_full)

# Helper function to create price volatility (lag n) features 
def calculate_vol_n(row, n, df, price_name):
    if row['type'] in ['mint', 'burn']:
        mint_burn_rows = df[df['type'].isin(['mint', 'burn'])]
        current_index = mint_burn_rows[mint_burn_rows.index <= row.name].index[-1]
        
        # Check if there are enough previous "mint" or "burn" rows
        if len(mint_burn_rows[mint_burn_rows.index < current_index]) >= n:
            prices = mint_burn_rows[mint_burn_rows.index <= current_index].tail(n+1)[price_name]
            return prices.std()
    return None

# Adds Price & Market Depth Features to the input DataFrame
def price_liquidity_features(df, main_is_5 = True):
    df_with_features = df
    
    # Market depth features
    df_with_features["depth_diff"] = df_with_features["snapshot_30"] - df_with_features["snapshot_5"]
    df_with_features["depth_ratio"] = df_with_features["snapshot_30"] / df_with_features["snapshot_5"]
    
    lags = [1, 2, 3]
    
    # Price features
    if main_is_5:
        df_with_features["price_diff"] = df_with_features["price_5"] - df_with_features["price_30"]
        for n in lags:
            df_with_features[f'vol{n}'] = df_with_features.apply(calculate_vol_n, args=(n, df_with_features, "price_5"), axis=1)
    else:
        df_with_features["price_diff"] = df_with_features["price_30"] - df_with_features["price_5"]
        for n in lags:
            df_with_features[f'vol{n}'] = df_with_features.apply(calculate_vol_n, args=(n, df_with_features, "price_30"), axis=1)
    
    df_with_features.drop(columns=['price_5', 'snapshot_5', 'price_30', 'snapshot_30'], inplace=True)
    
    df_with_features = df_with_features[(df_with_features["type"] == "mint") | (df_with_features["type"] == "burn")]
    df_with_features.reset_index(drop=True, inplace=True)
    
    return df_with_features

# Add Price & Market Depth Features to the Pool 5 DataFrame
usdc_weth_5_processed = price_liquidity_features(usdc_weth_5_events_snapshots, main_is_5 = True)

# Remove observations where a feature or target is NaN and reset index for the Pool 5 DataFrame
usdc_weth_5_ready = usdc_weth_5_processed.dropna(subset = ["s3_mint_other", "next_burn_time_other"])
usdc_weth_5_ready.reset_index(drop=True, inplace=True)

# Save the final Pool 5 DataFrame in CSV format 
usdc_weth_5_ready.to_csv("usdc_weth_5_final.csv", index = False)


""" USDC-WETH-0.003 Dataset """

""" Lagged Main-Pool Features """

# Add Lagged Main-Pool Features to the Pool 30 DataFrame
lags = [1, 2, 3]
event_types = ["mint", "burn"]
for e_type in event_types:
    for lag in lags:
        usdc_weth_30_block = lagged_main_features(usdc_weth_30_block, n = lag, event_type = e_type)


"""Lagged Other-Pool Features """

# Concatenate events from Pool 5 to the Pool 30 DataFrame and reset index
# Place events in Pool 5 first to break block_number ties so they are not considered future events
usdc_weth_30_concat = pd.concat([usdc_weth_30_block, usdc_weth_5_block]).sort_values(by=['block_number', 'pool_name'], ascending=[True, True])
usdc_weth_30_concat.reset_index(drop=True, inplace=True)

# Add Lagged Other-Pool Features to the Pool 30 DataFrame
lags = [1, 2, 3]
event_types = ["mint", "burn"]
for e_type in event_types:
    for lag in lags:
        usdc_weth_30_concat = lagged_other_features(usdc_weth_30_concat, n = lag, event_type = e_type, other_pool = 'USDC-WETH-0.0005')


""" Classification & Regression Targets """

# Add 2 classification targets to the Pool 30 DataFrame
usdc_weth_30_class = classification_targets(usdc_weth_30_concat, main_pool = 'USDC-WETH-0.003', other_pool = 'USDC-WETH-0.0005')

# Add 4 regression targets to the Pool 30 DataFrame
usdc_weth_30_targets = regression_targets(usdc_weth_30_class,  main_pool = 'USDC-WETH-0.003', other_pool = 'USDC-WETH-0.0005')


""" Price & Market Depth Features """

# Remove Pool 5 liquidity events from the Pool 30 DataFrame and reset index
usdc_weth_30_targets = usdc_weth_30_targets[usdc_weth_30_targets['pool_name'] == 'USDC-WETH-0.003']
usdc_weth_30_targets.reset_index(drop=True, inplace=True)

# Add snapshot (non-event) observations to the Pool 30 DataFrame
usdc_weth_30_full = non_event_blocks(usdc_weth_30_targets)

# Add price and market depth columns from both pools to the Pool 30 DataFrame
usdc_weth_30_events_snapshots = merged_snapshots(usdc_weth_30_full)

# Add Price & Market Depth Features to the Pool 30 DataFrame
usdc_weth_30_processed = price_liquidity_features(usdc_weth_30_events_snapshots, main_is_5 = False)

# Remove observations where a feature or target is NaN and reset index for the Pool 30 DataFrame
usdc_weth_30_ready = usdc_weth_30_processed.dropna(subset = ["s3_mint_main", "next_burn_time_main"])
usdc_weth_30_ready.reset_index(drop=True, inplace=True)

# Save the final Pool 30 DataFrame in CSV format
usdc_weth_30_ready.to_csv("usdc_weth_30_final.csv", index = False)
