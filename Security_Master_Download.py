# Import packages
import numpy as np
import pandas as pd
import xlwings as xw
import os
import time

# Set file paths
holdings_csv_path = 'Output/holdings.csv'
sec_master_path = 'Output/security_master.csv'
managers_path = 'Input/managers.xlsx'

# Import the holdings data from CSV
print(f"Loading holdings data from {holdings_csv_path}")
holdings_df = pd.read_csv(holdings_csv_path)
holdings_df = holdings_df.dropna(subset=['security_CIQ'])

# Get unique security IDs from holdings data
all_securities = holdings_df['security_CIQ'].unique()
print(f"Found {len(all_securities)} unique securities in holdings data")

# Check if security master file exists and load it
if os.path.exists(sec_master_path):
    print(f"Loading existing security master data from {sec_master_path}")
    existing_sec_master = pd.read_csv(sec_master_path)
    existing_sec_master = existing_sec_master.dropna(subset=['security_CIQ'])
    existing_securities = existing_sec_master['security_CIQ'].unique()
    print(f"Found {len(existing_securities)} securities in existing security master")

    # Identify new securities that need to be processed
    new_securities = np.setdiff1d(all_securities, existing_securities)
    print(f"Identified {len(new_securities)} new securities to process")
else:
    print(f"No existing security master file found at {sec_master_path}")
    new_securities = all_securities
    existing_sec_master = pd.DataFrame(columns=['security_CIQ', 'Name', 'ISIN', 'Industry', 'is_biotech', 'Ticker'])
    print(f"Will process all {len(new_securities)} securities")

# If there are no new securities, we can exit early
if len(new_securities) == 0:
    print("No new securities to process. Security master is up to date.")
    exit()

# Import spreadsheet
try:
    wb = xw.Book('Input/download.xlsx')  # Adjust the path if needed
    download = wb.sheets('Sheet1')
    
    # Set calculation to manual for performance
    wb.app.calculation = 'manual'
    
    # Clear the download sheet for reuse
    download.clear()
    
    # Process securities in batches to avoid Excel limitations
    batch_size = 100
    total_batches = (len(new_securities) + batch_size - 1) // batch_size
    
    for batch_num in range(total_batches):
        start_idx = batch_num * batch_size
        end_idx = min((batch_num + 1) * batch_size, len(new_securities))
        batch_securities = new_securities[start_idx:end_idx]
        
        print(f"Processing batch {batch_num + 1}/{total_batches} ({len(batch_securities)} securities)")
        
        # Start timing the batch processing
        start_time = time.time()
        
        # Reset row counter for each batch
        rowCount = 1
        
        # Set up formulas for security data
        for i, security_id in enumerate(batch_securities):
            Name = f"=CIQ(\"{security_id}\", \"IQ_COMPANY_NAME\")"
            isin = f"=CIQ(\"{security_id}\", \"IQ_ISIN\")"
            Industry = f"=CIQ(\"{security_id}\", \"IQ_PRIMARY_INDUSTRY\")"
            
            download.range('A' + str(rowCount)).value = security_id
            download.range('B' + str(rowCount)).value = Name
            download.range('C' + str(rowCount)).value = isin
            download.range('D' + str(rowCount)).value = Industry
            
            rowCount += 1
        
        # Calculate formulas
        wb.app.calculate()
        
        # Extract data into dataframe
        batch_data = download.range('A1:D' + str(rowCount-1)).value
        batch_df = pd.DataFrame(batch_data, columns=['security_CIQ', 'Name', 'ISIN', 'Industry'])
        
        # Add is_biotech column with default value False
        batch_df['is_biotech'] = False
        
        # Add Ticker column with empty values
        batch_df['Ticker'] = ''
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        minutes, seconds = divmod(elapsed_time, 60)
        print(f"Processed batch in {int(minutes)} minutes and {int(seconds)} seconds")
        
        # Append to the security master file if we have valid data
        if len(batch_df) > 0:
            # Append to the CSV file, with header only if the file doesn't exist yet
            batch_df.to_csv(sec_master_path, mode='a', header=not os.path.exists(sec_master_path), index=False)
            print(f"Appended {len(batch_df)} securities to {sec_master_path}")
        
        # Clear the download sheet for next batch
        download.clear()
    
    # Reset Excel calculation to automatic
    wb.save()
    wb.close()
    
    print(f"Security master data updated successfully at {sec_master_path}")
    
except Exception as e:
    print(f"Error processing security master data: {e}")
    # Ensure Excel is reset even if there's an error
    try:
        download.clear()
        wb.save()
        wb.close()
    except:
        pass

# Section 2: add a Biotechnology flag to the security master
print("\nStarting biotech classification process...")

# Load the updated security master
sec_master_df = pd.read_csv(sec_master_path)
manager_df = pd.read_excel(managers_path)

# First pass: Mark securities based on industry
biotech_industries = ['Biotechnology', 'Pharmaceuticals', 'Life Sciences Tools and Services']
sec_master_df['is_biotech'] = False
sec_master_df.loc[sec_master_df['Industry'].isin(biotech_industries), 'is_biotech'] = True

# Load holdings data for manager analysis
holdings_df = pd.read_csv(holdings_csv_path)
holdings_df = holdings_df[(holdings_df['shares'] > 1) & 
                          (holdings_df['value'] > 0) & 
                          (holdings_df['value'] < 100000)]
holdings_df = holdings_df.dropna(subset=['security_CIQ'])

holdings_df = holdings_df.merge(
    manager_df[['CIQ', 'Master']],
    left_on='holder_CIQ',
    right_on='CIQ',
    how='left'
)

# Merge holdings with security master to get biotech flags
holdings_df = holdings_df.merge(
    sec_master_df[['security_CIQ', 'is_biotech']],
    on='security_CIQ',
    how='left'
)

holdings_df['is_biotech'] = holdings_df['is_biotech'].fillna(False)

# Calculate biotech percentage for each manager using groupby operations
manager_biotech_pct = holdings_df.groupby('Master').apply(
    lambda x: (x[x['is_biotech']]['value'].sum() / x['value'].sum() * 100) if x['value'].sum() > 0 else 0
)

# Print biotech percentages with better formatting to ensure correct display
print("Biotech allocation by manager:")
print("-" * 40)
for manager in manager_biotech_pct.index:
    biotech_pct = manager_biotech_pct[manager]
    print(f"{manager:<30}: {biotech_pct:>6.2f}%")
print("-" * 40)

# Identify biotech managers (75% threshold)
biotech_managers = manager_biotech_pct[manager_biotech_pct >= 75].index.tolist()
print(f"Identified {len(biotech_managers)} biotech managers")

# Get securities owned by biotech managers
biotech_manager_holdings = holdings_df[holdings_df['Master'].isin(biotech_managers)]

# Count how many biotech managers own each security
manager_counts = biotech_manager_holdings.groupby('security_CIQ')['Master'].nunique().reset_index()
manager_counts.columns = ['security_CIQ', 'biotech_manager_count']

# Filter for securities owned by at least two biotech managers
securities_with_multiple_managers = manager_counts[manager_counts['biotech_manager_count'] >= 2]['security_CIQ'].tolist()
biotech_manager_securities = securities_with_multiple_managers

# Find asset management securities owned by biotech managers
asset_mgmt_securities = sec_master_df[
    (sec_master_df['Industry'] == 'Asset Management and Custody Banks') &
    (sec_master_df['security_CIQ'].isin(biotech_manager_securities))
]

# Filter out ETFs, Funds, Trusts, and ETNs
asset_mgmt_securities = asset_mgmt_securities[
    ~asset_mgmt_securities['Name'].str.contains(' ETF| Fund| Trust| ETN', case=False, na=False)
]

# Mark these securities as biotech
sec_master_df.loc[sec_master_df['security_CIQ'].isin(asset_mgmt_securities['security_CIQ']), 'is_biotech'] = True

# Print asset management securities that were marked as biotech
print("\nAsset Management Securities Marked as Biotech:")
print("-" * 80)
print(f"{'Name':<50} {'CIQ ID':<15} {'Industry':<30}")
print("-" * 80)
for _, row in asset_mgmt_securities.iterrows():
    print(f"{row['Name']:<50} {row['security_CIQ']:<15} {row['Industry']:<30}")
print("-" * 80)

# Save the updated security master
sec_master_df.to_csv(sec_master_path, index=False)
print(f"Updated security master with biotech classifications saved to {sec_master_path}")

# Print summary statistics
total_securities = len(sec_master_df)
biotech_securities = sec_master_df['is_biotech'].sum()
print(f"\nBiotech Classification Summary:")
print(f"Total securities: {total_securities}")
print(f"Biotech securities: {biotech_securities}")

# Print securities that are flagged as biotech but have blank tickers
biotech_no_ticker = sec_master_df[(sec_master_df['is_biotech'] == True) & 
                                  (sec_master_df['Ticker'].isna() | (sec_master_df['Ticker'] == ''))]
if len(biotech_no_ticker) > 0:
    print(f"\nFound {len(biotech_no_ticker)} biotech securities with missing tickers:")
    print("-" * 80)
    print(f"{'Name':<50} {'CIQ ID':<15} {'Industry':<30}")
    print("-" * 80)
    for _, row in biotech_no_ticker.iterrows():
        print(f"{row['Name']:<50} {row['security_CIQ']:<15} {row['Industry']:<30}")
    print("-" * 80)
else:
    print("\nAll biotech securities have ticker information.")