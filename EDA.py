# Required Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import gzip
import os 
import json

# Plot style
sns.set(style="darkgrid")

# Turn a string into a list of dictionaries to access individual entries
def parse_json(entry):
    valid_json = entry.replace("'", "\"")
    return json.loads(valid_json)


""" Pool Reference Data """

# Open pool reference data as a DataFrame
with gzip.open('full_pool_reference/pool') as f:
    reference = pd.read_csv(f)


# Change column data types to plot categories
reference['protocol'] = reference['protocol'].astype('category')
reference['fee'] = reference['fee'].astype('category')

# Create pool reference data plots
def plot_reference_info(categ, title):
    plt.figure(figsize=(10, 6), dpi = 300)
    
    # Calculate the counts of each category
    categ_counts = reference[categ].value_counts()
    
    # Get the top 7 categories by counts
    top_7_categ = categ_counts.head(7).index
    
    # Create a countplot with the top 7 categories
    ax = sns.countplot(data=reference, x=categ, order=top_7_categ)
    
    # Add labels and a title
    plt.xlabel(categ, fontsize=14)
    plt.ylabel("Count", fontsize=14)
    plt.title(title, fontsize=22, y=1.06)
    
    # X-axis tick label font size
    ax.tick_params(axis='x', labelsize=14)

    # Dertical adjustment for the annotations
    vertical_adjustment = 5  

    # Add counts above each bar 
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height() + vertical_adjustment),
                    ha='center', va='bottom', fontsize=12, color='black', weight='bold')

    plt.show()

# Figure 3.1
plot_reference_info('protocol', title="Distribution of Top 7 Liquidity Pool Protocols (Reference)")

# Figure 3.2
plot_reference_info('fee', title="Distribution of Top 7 Liquidity Pool Fee Tiers (Reference)")


""" Liquidity Events Data """

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

# Table 3.1
full_events[(full_events['pool_name'] == 'USDC-WETH-0.0005') | (full_events['pool_name'] == 'USDC-WETH-0.003')].head(20)

# Table 3.2
full_events['exchange'].value_counts()

# Table 3.3
full_events['pool_name'].value_counts().head(8)

# Create a DataFrame counting daily mints and burns for a specified pool
def extract_daily_events(folder_name, pool_name):
    daily_liquidity_events = pd.DataFrame()
    
    # Get a list of all files within the liquidity events folder
    file_list = os.listdir(folder_name)
    
    # Iterate over all but the last file within the "full_liquidity_events" folder
    # The June 29th file is incomplete, so avoid considering it
    for idx, file in enumerate(file_list[:-1]):
        with gzip.open("full_liquidity_events/" + file) as f:
            daily_events = pd.read_csv(f)
        daily_events_pool = daily_events[daily_events['pool_name'] == pool_name]
        
        # Daily mint and burn aggregation
        if not daily_events_pool.empty:
            daily_mints = daily_events_pool[daily_events_pool['type'] == 'mint'].shape[0]
            daily_burns = daily_events_pool[daily_events_pool['type'] == 'burn'].shape[0]
            
        else: 
            daily_mints = 0
            daily_burns = 0

        # Add the daily results to the DataFrame    
        daily_data = [file, pool_name, daily_mints, daily_burns]
        daily_df = pd.DataFrame([daily_data], columns=['Date', 'Pool', 'Mints', 'Burns'])
        daily_liquidity_events = pd.concat([daily_liquidity_events, daily_df], axis=0)
    
    # Sort by date and reset index
    daily_liquidity_events['Date'] = pd.to_datetime(daily_liquidity_events['Date'], format="%Y%m%d")
    daily_liquidity_events.sort_values(by='Date', inplace=True)
    daily_liquidity_events.reset_index(drop=True, inplace=True)
    
    return daily_liquidity_events

# Daily mints and burns DataFrame for pool USDT-BUSD
usdt_busd_events = extract_daily_events("full_liquidity_events", "USDT-BUSD")

# Plot daily mints and burns for the input DataFrame
def plot_daily_events(data, title): 
    plt.figure(figsize=(12, 6), dpi=300) 

    # Create a lineplot for mints
    sns.lineplot(data=data, x='Date', y='Mints', label='Mints', color='green')

    # Add a line plot for burns
    sns.lineplot(data=data, x='Date', y='Burns', label='Burns', color='red')

    # Add labels and a legend
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.title(title, fontsize=26, y=1.06)
    plt.legend(fontsize=16)
    
    # Set axis tick font sizes
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.show()

# Figure 3.3
plot_daily_events(usdt_busd_events, title='Daily Mints and Burns on the USDT-BUSD PancakeSwap Pool')

# Daily mints and burns DataFrame for pool DRIP-BUSD
drip_busd_events = extract_daily_events("full_liquidity_events", "DRIP-BUSD")

# Figure 3.4 
plot_daily_events(drip_busd_events, title='Daily Mints and Burns on the DRIP-BUSD PancakeSwap Pool')


""" Pool Snapshots Data """

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

# Calculate the USD size of market depth
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

# Table 3.4
usdc_weth_5_snapshots.head(2)


""" Uniswap v3 Liquidity Events """

# Filter the liquidity events DataFrame for USP3 events
usp3_events = full_events[full_events['exchange'] == 'usp3'].copy()

# Table 3.5
usp3_events['pool_name'].value_counts().head(10)

# 
usp3_events['pool_name'].value_counts().head(10).sum()

# Daily mints and burns DataFrames for USDC-WETH-0.0005 and USDC-WETH-0.003
usdc_weth_5_events = extract_daily_events("full_liquidity_events", "USDC-WETH-0.0005")
usdc_weth_30_events = extract_daily_events("full_liquidity_events", "USDC-WETH-0.003")

# Plot daily mints and burns for the input DataFrame with annotations  
def plot_daily_events_annot(data, title, y_coord=0):
    plt.figure(figsize=(12, 6), dpi=300) 

    # Create a lineplot for mints
    sns.lineplot(data=data, x='Date', y='Mints', label='Mints', color='green')

    # Add a line plot for burns 
    sns.lineplot(data=data, x='Date', y='Burns', label='Burns', color='red')
    
    # Find the date corresponding to the peak in burns
    max_burns_date = data.loc[data['Burns'].idxmax()]['Date']

    # Find the date corresponding to the peak in mints
    max_mints_date = data.loc[data['Mints'].idxmax()]['Date']
    
    # Annotate the peak in burns
    plt.annotate(f'{max_burns_date.strftime("%Y-%m-%d")}', xy=(max_burns_date, data['Burns'].max()), xytext=(-100, 0),
             textcoords='offset points', arrowprops=dict(arrowstyle='->', color='red'))
    
    # Annotate the peak in mints
    plt.annotate(f'{max_mints_date.strftime("%Y-%m-%d")}', xy=(max_mints_date, data['Mints'].max()), xytext=(-100, y_coord),
             textcoords='offset points', arrowprops=dict(arrowstyle='->', color='green'))
    
    # Add labels and a legend
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.title(title, fontsize=24, y=1.06)
    plt.legend(fontsize=16)
    
    # Set axis tick font sizes
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.show()

# Figure 3.5 
plot_daily_events_annot(usdc_weth_5_events, title='Daily Mints and Burns on the USDC-WETH-0.0005 USP3 Pool')

# Figure 3.7 
plot_daily_events_annot(usdc_weth_30_events, title='Daily Mints and Burns on the USDC-WETH-0.003 USP3 Pool',
                           y_coord=20)


# Get USD size of a liquidity event
def usd_volume(row):
    usdc = row['s0_usdc'] if not pd.isnull(row['s0_usdc']) else 0
    weth = row['s0_weth'] if not pd.isnull(row['s0_weth']) else 0
    return (usdc * 1) + (weth / row['price'])

# Create a DataFrame measuring daily mint and burn USD sizes for the input pool
def extract_event_sizes_usd(folder_name, pool_name):
    daily_liquidity_events = pd.DataFrame()
    
    # Get a list of all files within the liquidity events folder
    file_list = os.listdir(folder_name)
    
    # Iterate over all but the last file within the "full_liquidity_events" folder
    # The June 29th file is incomplete, so avoid considering it
    for idx, file in enumerate(file_list[:-1]):
        with gzip.open("full_liquidity_events/" + file) as f:
                daily_events = pd.read_csv(f)
        
        daily_events_pool = daily_events[daily_events['pool_name'] == pool_name].copy()
        
        # Aggregate daily mints and burns and measure them in USD
        if not daily_events_pool.empty:
            daily_events_pool['amounts'] = daily_events_pool['amounts'].apply(parse_json)
            daily_events_pool['s0_usdc'] = daily_events_pool['amounts'].apply(lambda x: x[0].get('amount') if x else None)
            daily_events_pool['s0_weth'] = daily_events_pool['amounts'].apply(lambda x: x[1].get('amount') if x else None)
            daily_events_pool.dropna(subset=['s0_usdc', 's0_weth'], how = 'all', inplace = True)
            daily_events_pool['s0_usd'] = daily_events_pool.apply(usd_volume, axis = 1)
            
            daily_mint_size = daily_events_pool.loc[daily_events_pool['type'] == 'mint', 's0_usd'].sum()
            daily_burn_size = daily_events_pool.loc[daily_events_pool['type'] == 'burn', 's0_usd'].sum()
            
        else:
            daily_mint_size = 0
            daily_burn_size = 0
        
        # Add daily data to the final DataFrame
        daily_data = [file, pool_name, daily_mint_size, daily_burn_size]
        daily_df = pd.DataFrame([daily_data], columns=['Date', 'Pool', 'Mint Size', 'Burn Size'])
        daily_liquidity_events = pd.concat([daily_liquidity_events, daily_df], axis=0)
    
    # Sort by date and reset index
    daily_liquidity_events['Date'] = pd.to_datetime(daily_liquidity_events['Date'], format="%Y%m%d")
    daily_liquidity_events.sort_values(by='Date', inplace=True)
    daily_liquidity_events.reset_index(drop=True, inplace=True)
    
    return daily_liquidity_events

# Daily mint and burn USD size DataFrames for USDC-WETH-0.0005 and USDC-WETH-0.003
usdc_weth_5_sizes = extract_event_sizes_usd("full_liquidity_events", "USDC-WETH-0.0005")
usdc_weth_30_sizes = extract_event_sizes_usd("full_liquidity_events", "USDC-WETH-0.003")

# Plot daily mint and burn USD sizes for the input pool 
def plot_event_sizes_usd(data, title): 
    plt.figure(figsize=(12, 6), dpi=300) 
    
    # Normalize USD sizes to millions
    data['Mint Size (Normalized)'] = data['Mint Size'] / 1e6  
    data['Burn Size (Normalized)'] = data['Burn Size'] / 1e6  
    
    # Lineplots for mint and burn USD sizes
    sns.lineplot(data=data, x='Date', y='Mint Size (Normalized)', label='Mint Size (Millions USD)', 
                color='green')
    
    sns.lineplot(data=data, x='Date', y='Burn Size (Normalized)', label='Burn Size (Millions USD)',
                color='red')
    
    # Find the date corresponding to the peak in burn size
    max_burns_date = data.loc[data['Burn Size (Normalized)'].idxmax()]['Date']

    # Find the date corresponding to the peak in mint size
    max_mints_date = data.loc[data['Mint Size (Normalized)'].idxmax()]['Date']

    # Annotate the peak in burn size
    plt.annotate(f'{max_burns_date.strftime("%Y-%m-%d")}', xy=(max_burns_date, data['Burn Size (Normalized)'].max()), xytext=(-100, 0),
             textcoords='offset points', arrowprops=dict(arrowstyle='->', color='red'))

    # Annotate the peak in mint size
    plt.annotate(f'{max_mints_date.strftime("%Y-%m-%d")}', xy=(max_mints_date, data['Mint Size (Normalized)'].max()), xytext=(-100, -50),
             textcoords='offset points', arrowprops=dict(arrowstyle='->', color='green'))

    # Add labels and a legend
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Millions of USD', fontsize=14)
    plt.title(title, fontsize=22, y=1.06)
    plt.legend(fontsize=16)
    
    # Set axis tick font sizes
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.show()

# Figure 3.6
plot_event_sizes_usd(usdc_weth_5_sizes, title='Daily Mint and Burn Sizes on the USDC-WETH-0.0005 USP3 Pool')

# Figure 3.8 
plot_event_sizes_usd(usdc_weth_30_sizes, title='Daily Mint and Burn Sizes on the USDC-WETH-0.003 USP3 Pool')

# Plot block-wise market depth for Pools 5 and 30
def plot_market_depth():
    plt.figure(figsize=(12, 6), dpi=300)
    
    # Normalize USD snapshots to thousands
    usdc_weth_5_snapshots['snapshot_5 (Thousands)'] = usdc_weth_5_snapshots['snapshot_5'] / 1e3
    usdc_weth_30_snapshots['snapshot_30 (Thousands)'] = usdc_weth_30_snapshots['snapshot_30'] / 1e3
    
    # Add lineplots of market depth for Pools 5 and 30
    sns.lineplot(data=usdc_weth_5_snapshots, x='date', y='snapshot_5 (Thousands)', label='Market Depth (Pool 5)')
    sns.lineplot(data=usdc_weth_30_snapshots, x='date', y='snapshot_30 (Thousands)', label='Market Depth (Pool 30)')
    
    # Add labels and a legend
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Thousands of USD', fontsize=14)
    plt.title('Block-Wise Market Depth on the USDC-WETH USP3 Sister Pools', fontsize=22, y=1.06)
    plt.legend(fontsize=16)
    
    # Set axis tick font sizes
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    plt.show()

# Figure 3.9
plot_market_depth()