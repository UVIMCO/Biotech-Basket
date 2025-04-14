import pandas as pd
import numpy as np

class FilingBacktester:
    def __init__(self, 
                 holdings_path='Output/holdings.csv',
                 sec_master_path='Output/security_master.csv',
                 sec_mapping_path='Output/security_mapping.csv',
                 sec_out_path='Output/security_out.csv',
                 managers_path='Input/managers.xlsx'):
        """
        Initialize the FundAnalyzer with necessary data files.
        
        Parameters:
        -----------
        holdings_path : str
            Path to the holdings CSV file
        sec_master_path : str
            Path to the security master CSV file
        sec_mapping_path : str
            Path to the security mapping CSV file
        sec_out_path : str
            Path to the security out CSV file
        managers_path : str
            Path to the managers Excel file
        """

        self.manager_df = pd.read_excel(managers_path)
        self.manager_df['Contamination'] = self.manager_df['Contamination'] == 1.0

        self.sec_master_df = pd.read_csv(sec_master_path)
        self.sec_master_df = self.sec_master_df.dropna(subset=['security_CIQ'])
        self.sec_master_df = self.sec_master_df[self.sec_master_df['is_biotech'] == True] # Filter for biotech securities

        self.sec_mapping_df = pd.read_csv(sec_mapping_path)
        self.sec_mapping_df['Min_Px_Date'] = pd.to_datetime(self.sec_mapping_df['Min_Px_Date'], format='%Y-%m-%d')
        self.sec_mapping_df['Max_Px_Date'] = pd.to_datetime(self.sec_mapping_df['Max_Px_Date'], format='%Y-%m-%d')
        self.sec_mapping_df['Ticker'] = self.sec_mapping_df['Ticker'].str[:-7] # Drop the " Equity" from the ticker

        self.sec_out_df = pd.read_csv(sec_out_path)
        self.sec_out_df['date'] = pd.to_datetime(self.sec_out_df['date'])
        self.sec_out_df['Ticker'] = self.sec_out_df['Ticker'].str[:-7] # Drop the " Equity" from the ticker

        self.holdings_df = pd.read_csv(holdings_path)
        self.holdings_df = self.holdings_df.dropna(subset=['security_CIQ'])
        self.holdings_df['holding_date'] = pd.to_datetime(self.holdings_df['holding_date'])
        self.holdings_df['position_date'] = pd.to_datetime(self.holdings_df['position_date'])
        self.holdings_df = self.holdings_df[(self.holdings_df['shares'] > 1) & 
                                          (self.holdings_df['value'] > 0) & 
                                          (self.holdings_df['value'] < 100000)]
        
        self.holdings_df = pd.merge(self.holdings_df, self.manager_df[['CIQ', 'Master']], left_on='holder_CIQ', right_on='CIQ', how='left')
        self.holdings_df = pd.merge(self.holdings_df, self.sec_master_df, on='security_CIQ', how='left')
        self.holdings_df = pd.merge(self.holdings_df, self.sec_mapping_df, on='Ticker', how='left')

        self.error_df = self.holdings_df[(self.holdings_df['PX_NA_Flag'] == True) |
                       (self.holdings_df['Max_Px_Date'] < self.holdings_df['holding_date']) |
                       (self.holdings_df['Min_Px_Date'] > self.holdings_df['holding_date'])]
        self.error_df = self.error_df[['holding_date', 'Master', 'Ticker', 'holding_date', 'value', 'PX_NA_Flag', 'Max_Px_Date', 'Min_Px_Date']]

        self.holdings_df = self.holdings_df[(self.holdings_df['PX_NA_Flag'] == False) &
                       (self.holdings_df['Max_Px_Date'] >= self.holdings_df['holding_date']) &
                       (self.holdings_df['Min_Px_Date'] <= self.holdings_df['holding_date'])]
        self.holdings_df = self.holdings_df[['holding_date', 'Master', 'Ticker', 'value', 'Name']]
