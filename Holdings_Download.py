# Import packages
import numpy as np
import pandas as pd
import xlwings as xw
from datetime import date
import time
import os

# Set Global Variables
startDate = pd.Timestamp('2014-12-31').date()
endDate = pd.Timestamp('2024-12-31').date()
currency = "USD"
holdings_path = 'Output/holdings.csv'
holdings_index_path = 'Output/holdings_index.csv'

# import spreadsheet
input_wb = xw.Book('Input/managers.xlsx')
managerList = input_wb.sheets('List')
download_wb = xw.Book('Input/download.xlsx')
download = download_wb.sheets('Sheet1')

listDate = (pd.date_range(startDate, endDate + pd.offsets.QuarterBegin(1), freq = 'QE')
            .strftime('%Y-%m-%d')
            .tolist())

# Import manager list from the Excel file
manager_df = pd.DataFrame(managerList.range('A1').expand('table').value)

# Set the first row as column headers
headers = manager_df.iloc[0].tolist()
manager_df = manager_df.iloc[1:].reset_index(drop=True)
manager_df.columns = headers

input_wb.close()

#TEST
#manager_df = manager_df.head(10)

# Ensure column names are correct
if 'Master' in headers and 'CIQ' in headers and 'Legal' in headers and 'Max' in headers:
    print(f"Successfully imported manager list with {len(manager_df)} managers")
else:
    print("Warning: Expected columns not found in manager list")

# Load or create the holdings index
try:
    if os.path.exists(holdings_index_path):
        holdings_index = pd.read_csv(holdings_index_path)
        print(f"Loaded existing holdings index with {len(holdings_index)} entries")
    else:
        holdings_index = pd.DataFrame(columns=['holder_CIQ', 'holding_date'])
        holdings_index.to_csv(holdings_index_path, index=False)
        print("Created new holdings index file")
except Exception as e:
    print(f"Error loading holdings index: {e}")
    holdings_index = pd.DataFrame(columns=['holder_CIQ', 'holding_date'])

# Check if the output file already exists and load the list of processed managers
try:
    if os.path.exists(holdings_path):
        # Read the existing holdings file to get the list of managers already processed
        existing_holdings = pd.read_csv(holdings_path)
        existing_managers = existing_holdings['holder_CIQ'].unique()
        print(f"Found {len(existing_managers)} existing managers in {holdings_path}")
        
        # Get existing combinations of holder_CIQ and holding_date
        existing_combinations = existing_holdings[['holder_CIQ', 'holding_date']].drop_duplicates()

        # Find combinations that aren't in the index yet
        new_combinations = existing_combinations.merge(
            holdings_index,
            on=['holder_CIQ', 'holding_date'],
            how='left',
            indicator=True
        )
        new_combinations = new_combinations[new_combinations['_merge'] == 'left_only']
        new_combinations = new_combinations.drop('_merge', axis=1)
        
        if len(new_combinations) > 0:
            print(f"Adding {len(new_combinations)} existing combinations to holdings index")
            holdings_index = pd.concat([holdings_index, new_combinations], ignore_index=True)
            holdings_index.to_csv(holdings_index_path, index=False)
    else:
        print(f"No existing holdings file found at {holdings_path}. Will create a new file.")
except Exception as e:
    print(f"Error checking existing holdings: {e}")
    print("Continuing with all managers in the list.")

download_wb.app.calculation = 'manual'
# Sort manager_df by 'Max' column in ascending order
manager_df = manager_df.sort_values(by='Max', ascending=True)
manager_df = manager_df.reset_index(drop=True)

# formula to download holdings info from CIQ
for n in range(len(manager_df['CIQ'])):
    CIQ_ManagerID = manager_df['CIQ'][n]
    legal_name = manager_df['Legal'][n]
    max_holdings = int(manager_df['Max'][n])
    print(f"Beginning download process for {legal_name}")

    start_time = time.time()

    for i in range(len(listDate)):
        date = listDate[i]

        # Check if the combination exists without using set_index
        if holdings_index[(holdings_index['holder_CIQ'] == CIQ_ManagerID) & 
                          (holdings_index['holding_date'] == date)].shape[0] > 0:
            #print(f"Skipping {legal_name} for {date} - already processed")
            continue

        #clear the download sheet
        download.clear()

        security_CIQ = ("=CIQRANGE(\"" + CIQ_ManagerID + "\", \"IQ_HOLDING_CIQID\", 1, " + str(max_holdings) + ", \"" + date + "\"")
        Shares = ("=CIQRANGE(\"" + CIQ_ManagerID + "\", \"IQ_HOLDING_SHARES\", 1, " + str(max_holdings) + ", \"" + date + "\"")
        Value = ("=CIQRANGE(\"" + CIQ_ManagerID + "\", \"IQ_HOLDING_VALUE\", 1, " + str(max_holdings) + ", \"" + date + "\" ,,\"" + currency + "\"")
        Date = ("=CIQRANGE(\"" + CIQ_ManagerID + "\", \"IQ_HOLDING_POSITION_DATE\", 1, " + str(max_holdings) + ", \"" + date + "\"")

        download.range('A1').value = security_CIQ
        download.range('B1').value = Shares
        download.range('C1').value = Value
        download.range('D1').value = Date

        download_wb.app.calculate()

        cur_holdings = pd.DataFrame(download.range(f'A1:D{max_holdings+1}').value, columns=['security_CIQ', 'shares', 'value', 'position_date'])
        cur_holdings['holding_date'] = date
        cur_holdings['holder_CIQ'] = CIQ_ManagerID
        cur_holdings = cur_holdings[['holding_date', 'holder_CIQ', 'security_CIQ', 'shares', 'value', 'position_date']]

        # Convert data types to appropriate formats
        cur_holdings['shares'] = pd.to_numeric(cur_holdings['shares'], errors='coerce').astype('Int64')
        cur_holdings['value'] = pd.to_numeric(cur_holdings['value'], errors='coerce').astype(float)

        cur_holdings = cur_holdings.dropna(how='any')

        if(len(cur_holdings) == max_holdings):
            print(f"Warning: {legal_name} has {max_holdings} holdings on {date}. May be missing holdings.")

        cur_holdings = cur_holdings[cur_holdings['shares'] > 0]
        cur_holdings = cur_holdings[cur_holdings['value'] > 0]

        if(len(cur_holdings) > 0):
            cur_holdings['position_date'] = pd.to_datetime(cur_holdings['position_date'], errors='coerce', origin='1899-12-30', unit='D')
            cur_holdings = cur_holdings.dropna(how='any')
            cur_holdings['position_date'] = cur_holdings['position_date'].dt.strftime('%Y-%m-%d')

        # Print completion message with statistics about the holdings
        num_holdings = len(cur_holdings)
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        minutes, seconds = divmod(elapsed_time, 60)
        print(f"Completed download for {legal_name} on {date} after {int(minutes)} minutes and {int(seconds)} seconds with {num_holdings} holdings")

        # Add this manager-date combination to the index
        new_index_entry = pd.DataFrame({
            'holder_CIQ': [CIQ_ManagerID],
            'holding_date': [date]
        })
        holdings_index = pd.concat([holdings_index, new_index_entry], ignore_index=True)
        holdings_index.to_csv(holdings_index_path, index=False)

        if num_holdings > 0:
            # Append to the CSV file after each manager if we have valid holdings
            cur_holdings.to_csv(holdings_path, mode='a', header=not os.path.exists(holdings_path), index=False)

download.clear()
download_wb.save()
download_wb.close()

