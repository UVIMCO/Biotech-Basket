import pandas as pd
import numpy as np
import statsmodels.api as sm

def get_top_pct(row, pct=0.999):
    # Drop NaN values
    valid_values = row.dropna()
    if valid_values.empty:
        return pd.Series(False, index=row.index)
    
    # Sort values in descending order
    sorted_values = valid_values.sort_values(ascending=False)
    
    # Calculate cumulative percentage
    cumulative_pct = sorted_values.cumsum() / sorted_values.sum()
    
    # Create a mask for values within the threshold
    mask = cumulative_pct <= pct
    
    # Add one more item to go over the threshold
    # Get the count of True values
    true_count = mask.sum()
    
    # If we have items and haven't included everything already
    if true_count < len(sorted_values):
        # Add one more item (the next one after the cutoff)
        mask.iloc[true_count] = True
    
    # Create result series with the same index as the input row
    result = pd.Series(False, index=row.index)
    result[mask.index] = mask
    
    return result

def get_top_n(row, n=100):
    
    valid_values = row.dropna()
    if valid_values.empty:
        return pd.Series(False, index=row.index)
    
    sorted_values = valid_values.sort_values(ascending=False)
    
    top_n_values = sorted_values.head(n)
    
    result = pd.Series(False, index=row.index)
    result[top_n_values.index] = True
    
    return result

class BacktesterData:
    def __init__(self, 
                 holdings_path='Output/holdings.csv',
                 sec_master_path='Output/security_master.csv',
                 sec_mapping_path='Output/security_mapping.csv',
                 sec_out_path='Output/security_out.csv',
                 managers_path='Input/managers.xlsx'):
        """
        Initialize the BacktesterData with necessary data files.
        
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

        # Manager Data
        self.manager_df = pd.read_excel(managers_path)
        self.manager_df['Contamination'] = self.manager_df['Contamination'] == 1.0
        self.manager_df['Not_Biotech'] = self.manager_df['Not_Biotech'] == 1.0

        # Security Master Data from CIQ
        self.sec_master_df = pd.read_csv(sec_master_path)
        self.sec_master_df = self.sec_master_df.dropna(subset=['security_CIQ'])
        self.sec_master_df = self.sec_master_df[self.sec_master_df['is_biotech'] == True] # Filter for biotech securities

        # Security Mapping Data (from BBG bulk pull)
        self.sec_mapping_df = pd.read_csv(sec_mapping_path)
        self.sec_mapping_df['Min_Px_Date'] = pd.to_datetime(self.sec_mapping_df['Min_Px_Date'], format='%Y-%m-%d')
        self.sec_mapping_df['Max_Px_Date'] = pd.to_datetime(self.sec_mapping_df['Max_Px_Date'], format='%Y-%m-%d')
        self.sec_mapping_df['Ticker'] = self.sec_mapping_df['Ticker'].str[:-7] # Drop the " Equity" from the ticker

        # Holdings Data long format from CIQ
        self.holdings_df = pd.read_csv(holdings_path)
        self.holdings_df = self.holdings_df.dropna(subset=['security_CIQ'])
        self.holdings_df['holding_date'] = pd.to_datetime(self.holdings_df['holding_date'], format='%Y-%m-%d')
        self.holdings_df['position_date'] = pd.to_datetime(self.holdings_df['position_date'], format='%Y-%m-%d')
        self.holdings_df = self.holdings_df[(self.holdings_df['shares'] > 1) & 
                                          (self.holdings_df['value'] > 0) & 
                                          (self.holdings_df['value'] < 100000)]
        
        self.holdings_df = pd.merge(self.holdings_df, self.manager_df[['CIQ', 'Master']], left_on='holder_CIQ', right_on='CIQ', how='left')
        self.holdings_df = pd.merge(self.holdings_df, self.sec_master_df, on='security_CIQ', how='left')
        self.holdings_df = pd.merge(self.holdings_df, self.sec_mapping_df, on='Ticker', how='left')
        self.holdings_df = self.holdings_df.dropna(subset=['Ticker'])

        self.error_df = self.holdings_df[(self.holdings_df['Max_Px_Date'] < self.holdings_df['holding_date']) |
                       (self.holdings_df['Min_Px_Date'] > self.holdings_df['holding_date']) |
                       (self.holdings_df['Max_Px_Date'].isna()) |
                       (self.holdings_df['Min_Px_Date'].isna())]
        self.error_df = self.error_df[['holding_date', 'Master', 'Ticker', 'value', 'Name', 'PX_NA_Flag', 'Max_Px_Date', 'Min_Px_Date']]

        self.holdings_df = self.holdings_df[(self.holdings_df['Max_Px_Date'] >= self.holdings_df['holding_date']) &
                       (self.holdings_df['Min_Px_Date'] <= self.holdings_df['holding_date'])]
        
        self.holdings_df = self.holdings_df.groupby(['holding_date', 'Master', 'Ticker']).agg({'value': 'sum'}).reset_index()
        self.holdings_df = self.holdings_df[['holding_date', 'Master', 'Ticker', 'value']]

        # Daily Security Data (from BBG bulk pull)
        self.sec_out_df = pd.read_csv(sec_out_path)
        self.sec_out_df['date'] = pd.to_datetime(self.sec_out_df['date'])
        self.sec_out_df['Ticker'] = self.sec_out_df['Ticker'].str[:-7] # Drop the " Equity" from the ticker
        
        self.sec_rets_df = self.sec_out_df.pivot(index='date', columns='Ticker', values='TOT_RETURN_INDEX_GROSS_DVDS')
        self.sec_rets_df = self.sec_rets_df.sort_index()
        self.sec_rets_df = self.sec_rets_df.replace(0, np.nan).ffill()
        self.sec_rets_df = self.sec_rets_df / self.sec_rets_df.shift(1) - 1
        self.sec_rets_df = self.sec_rets_df.fillna(0)
        self.sec_mkt_cap_df = self.sec_out_df.pivot(index='date', columns='Ticker', values='CUR_MKT_CAP')
        self.sec_mkt_cap_df = self.sec_mkt_cap_df.sort_index()
        
        # Set market cap to NA for specific indices
        indices_to_exclude = ['XBI US', 'IBB US', 'MSXXNCB', 'LSCIB']
        for idx in indices_to_exclude:
            if idx in self.sec_mkt_cap_df.columns:
                self.sec_mkt_cap_df[idx] = np.nan

        self.sec_vol_df = self.sec_out_df.pivot(index='date', columns='Ticker', values='PX_VOLUME')
        self.sec_vol_df = self.sec_vol_df.sort_index()

        self.sec_price_df = self.sec_out_df.pivot(index='date', columns='Ticker', values='PX_LAST')
        self.sec_price_df = self.sec_price_df.sort_index()

        self.value_traded_df = self.sec_vol_df.rolling(window=90, min_periods=1).median() * self.sec_price_df
        
        # Get eligible securities based on market cap and price criteria
        # Ensure price is > $1.00 based on max price over past 30 days
        self.eligible_securities_df = self.sec_mkt_cap_df.shift(1).apply(get_top_pct, axis=1) & (self.sec_price_df.rolling(window=30, min_periods=1).max().shift(1) > 1.00)

class FilingBacktester:
    def __init__(self, data):
        
        # Manager Data
        self.manager_df = data.manager_df

        # Holdings Data long format
        self.holdings_df = data.holdings_df

        # Daily Security Data
        self.sec_rets_df = data.sec_rets_df
        self.sec_mkt_cap_df = data.sec_mkt_cap_df
        self.sec_vol_df = data.sec_vol_df

        # Derived from daily security data
        self.eligible_securities_df = data.eligible_securities_df
        self.value_traded_df = data.value_traded_df

    def get_fund_holdings(self, fund_name, eligible_securities=True, normalize=True, contamination=False, start_date=None, end_date=None):        
        
        if start_date is None:
            start_date = self.manager_df.loc[self.manager_df['Master'] == fund_name, 'Public_Start'].values[0]
        if end_date is None:
            end_date = self.manager_df.loc[self.manager_df['Master'] == fund_name, 'Public_End'].values[0]

        holdings_df = self.holdings_df[self.holdings_df['Master'] == fund_name]

        # Check if start_date is not None and not NaT/NaN
        if start_date is not None and pd.notna(start_date):
            holdings_df = holdings_df[holdings_df['holding_date'] >= start_date]
        # Check if end_date is not None and not NaT/NaN
        if end_date is not None and pd.notna(end_date):
            holdings_df = holdings_df[holdings_df['holding_date'] <= end_date]

        holdings_df = holdings_df.groupby(['holding_date', 'Ticker']).agg({'value': 'sum'}).reset_index()
        holdings_df = holdings_df.pivot(index='holding_date', columns='Ticker', values='value')
        holdings_df = holdings_df.sort_index()

        eligible_securities = self.eligible_securities_df.reindex(index=holdings_df.index, columns=holdings_df.columns)

        holdings_df = holdings_df * eligible_securities
        holdings_df = holdings_df.fillna(0)


        if contamination:
            mkt_cap = self.sec_mkt_cap_df.reindex(index=holdings_df.index, columns=holdings_df.columns)
            mean_ownership = holdings_df.sum(axis=1)/mkt_cap.sum(axis=1)
            holdings_df = holdings_df.sub(mkt_cap.multiply(mean_ownership, axis=0)).clip(lower=0)
  
        if normalize:
            holdings_df = holdings_df.div(holdings_df.sum(axis=1), axis=0)
            holdings_df = holdings_df.fillna(0)

        return holdings_df
        
    def adjust_holdings(self, holdings_df, mvs=False, max_participation=None, port_size=None, pct=None, n=None):

        if max_participation is not None and port_size is not None:
            value_traded = self.value_traded_df.reindex(index=holdings_df.index, columns=holdings_df.columns)
            max_position_size = value_traded.multiply(max_participation / port_size)

            adjusted_holdings_df = holdings_df.copy()
            fixed_mask_df = pd.DataFrame(False, index=holdings_df.index, columns=holdings_df.columns)
            num_securities = holdings_df.shape[1]
            converged = False
            for iteration in range(num_securities):

                over_limit_mask_df = (adjusted_holdings_df > max_position_size) & (~fixed_mask_df)
                
                if not over_limit_mask_df.any().any():
                    converged = True
                    break
                    
                # Fix any positions over the limit
                adjusted_holdings_df = adjusted_holdings_df.where(~over_limit_mask_df, max_position_size)

                # Get the remaining budget needed to be filled by non-fixed securities
                fixed_mask_df = fixed_mask_df | over_limit_mask_df
                fixed_sums = (adjusted_holdings_df * fixed_mask_df).sum(axis=1)
                remaining_budgets = (1.0 - fixed_sums).clip(lower=0)

                # Get the sum of the non-fixed securities
                non_fixed_sums = (adjusted_holdings_df * ~fixed_mask_df).sum(axis=1)
                
                scaling_factors = remaining_budgets / non_fixed_sums.replace(0, np.nan)
                # Handle NaN resulting from 0/0 or x/0
                scaling_factors = scaling_factors.fillna(0).infer_objects(copy=False)
                non_fixed_adjusted = adjusted_holdings_df.multiply(scaling_factors, axis=0)
                adjusted_holdings_df = adjusted_holdings_df.where(fixed_mask_df, non_fixed_adjusted).infer_objects(copy=False)

            if not converged:
                final_over_limit_mask_df = (adjusted_holdings_df > max_position_size + 1e-9) & (~fixed_mask_df)
                if final_over_limit_mask_df.any().any():
                     raise RuntimeError(f"Failed to converge after {num_securities} iterations. Constraints might be infeasible or numerical issues occurred.")

            final_row_sums = adjusted_holdings_df.sum(axis=1)
            min_sum_threshold = 0.99
            low_sum_mask = final_row_sums < min_sum_threshold
            
            if low_sum_mask.any():
                problematic_sums = final_row_sums[low_sum_mask]
                error_details = "\n".join([f"  Date: {date.strftime('%Y-%m-%d')}, Max Portfolio Capacity: {sum_val*port_size:.6f}" for date, sum_val in problematic_sums.items()])
                warning_message = (
                    f"Warning: Final portfolio weights sum to less than {min_sum_threshold} "
                    f"after iterative adjustment on some dates.\n"
                    f"Portfolio size was: {port_size} dollars.\n"
                    f"Problematic Dates and Actual Weight Sums:\n{error_details}"
                )
                print(warning_message)

            adjusted_holdings_df = adjusted_holdings_df.div(final_row_sums, axis=0)
            adjusted_holdings_df = adjusted_holdings_df.fillna(0)
            holdings_df = adjusted_holdings_df
        if pct is not None:
            top_pct = holdings_df.apply(get_top_pct, axis=1, pct=pct)
            holdings_df = holdings_df.multiply(top_pct)
        elif n is not None:
            top_n = holdings_df.apply(get_top_n, axis=1, n=n)
            holdings_df = holdings_df.multiply(top_n)

        if not mvs:
            row_sums = holdings_df.sum(axis=1)
            holdings_df = holdings_df.div(row_sums.replace(0, np.nan), axis=0).fillna(0)

        return holdings_df

    def get_strategy_returns(self, holdings_df, end_date):

        tickers = holdings_df.columns
        start_date = holdings_df.index.min()
        returns_df = self.sec_rets_df[tickers].loc[start_date:end_date]

        # Reindex holdings_df to match all dates in returns_df and fill forward
        holdings_df = holdings_df.reindex(returns_df.index)
        holdings_df = holdings_df.shift(1)
        holdings_df = holdings_df.ffill()
        holdings_df = holdings_df.fillna(0)
        
        # Multiply returns by aligned holdings and sum across securities
        strategy_returns = returns_df.multiply(holdings_df)
        strategy_returns = strategy_returns.sum(axis=1)

        return strategy_returns
    
    def get_strategy_attribution(self, holdings_df, end_date):

        tickers = holdings_df.columns
        start_date = holdings_df.index.min()
        returns_df = self.sec_rets_df[tickers].loc[start_date:end_date]

        # Reindex holdings_df to match all dates in returns_df and fill forward
        holdings_df = holdings_df.reindex(returns_df.index)
        holdings_df = holdings_df.shift(1)
        holdings_df = holdings_df.ffill()
        holdings_df = holdings_df.fillna(0)
        
        # Multiply returns by aligned holdings and sum across dates
        strategy_attribution = returns_df.multiply(holdings_df)

        return strategy_attribution
    
    def standardize_factor_scores(self, scores, clip_value=4):

        # BBG methodology

        mkt_cap = self.sec_mkt_cap_df.reindex(index=scores.index, columns=scores.columns)
        eligible = self.eligible_securities_df.reindex(index=scores.index, columns=scores.columns)
        
        mkt_cap = mkt_cap.multiply(eligible)
        mkt_cap = mkt_cap.replace(0, np.nan)

        median = scores.median(axis=1)
        robust_std_dev = 1.4826 * (scores.sub(median, axis=0).abs().median(axis=1))
        scores = scores.sub(median, axis=0).div(robust_std_dev, axis=0)
        scores = scores.clip(-clip_value, clip_value)

        mean = scores.mean(axis=1)
        std_dev = scores.std(axis=1)
        scores = scores.sub(mean, axis=0).div(std_dev, axis=0)

        weighted_mean = scores.mul(mkt_cap).sum(axis=1).div(mkt_cap.sum(axis=1))
        scores = scores.sub(weighted_mean, axis=0)

        return scores
    
    def get_factor_returns(self, factor_scores, clip_z=4):

        #Ensure that factor_scores for date n only use information from date n-1 and earlier

        valid_rets = (self.sec_rets_df == 0).sum(axis=1) < 0.9*len(self.sec_rets_df.columns)
        mkt_cap = self.sec_mkt_cap_df.multiply(self.eligible_securities_df)
        mkt_cap = mkt_cap.replace(0, np.nan)
        mkt_cap = mkt_cap.loc[valid_rets]
        rets = self.sec_rets_df.loc[valid_rets]

        factor_rets = pd.DataFrame(index=rets.index, columns=list(factor_scores.keys()) + ["Market"], dtype=np.float64)

        for i in range(1, len(rets)):
            date = rets.index[i]
            prev_date = rets.index[i-1]

            day_rets = rets.loc[date]
            prev_mkt_cap = mkt_cap.loc[prev_date]

            cur_factor_scores = {factor: factor_scores[factor].loc[date] for factor in factor_scores}
    
            mask = ~(np.isnan(day_rets.values) | np.isnan(prev_mkt_cap.values) | np.logical_or.reduce([np.isnan(cur_factor_scores[factor].values) for factor in cur_factor_scores]))

            day_rets = day_rets[mask]
            prev_mkt_cap = prev_mkt_cap[mask]
            cur_factor_scores = {factor: cur_factor_scores[factor][mask] for factor in cur_factor_scores}

            day_rets = day_rets.clip(day_rets.mean()-clip_z*day_rets.std(), day_rets.mean()+clip_z*day_rets.std())

            X = sm.add_constant(pd.DataFrame(cur_factor_scores))

            try:
                model = sm.WLS(day_rets, X, weights=prev_mkt_cap).fit()
            except Exception as e:
                print(f"Error fitting model for {date}: {e}")
                factor_rets.loc[date, list(factor_scores.keys())] = np.nan
                factor_rets.loc[date, "Market"] = np.nan
                continue

            factor_rets.loc[date, "Market"] = model.params['const']

            for factor in factor_scores.keys():
                factor_rets.loc[date, factor] = model.params[factor]

        return factor_rets
    
    def get_factor_attribution(self, holdings_df, factor_scores, factor_returns, end_date):

        #Get the factor attribution for a portfolio of holdings

        tickers = holdings_df.columns
        start_date = holdings_df.index.min()

        rets = self.sec_rets_df[tickers].loc[start_date:end_date]

        cur_factor_rets = factor_returns.loc[start_date:end_date]
        cur_factor_scores = {factor: factor_scores[factor].loc[start_date:end_date] for factor in factor_scores.keys()}

        port_factor_ctr = {factor: cur_factor_scores[factor].mul(cur_factor_rets[factor], axis=0).reindex(rets.index).astype(np.float64).fillna(0) for factor in factor_scores.keys()}
        port_factor_ctr['Market'] = pd.DataFrame(cur_factor_rets['Market'].values.reshape(-1, 1).repeat(len(tickers), axis=1), 
                                   index=cur_factor_rets.index, columns=tickers).reindex(rets.index).astype(np.float64).fillna(0)
        
        port_factor_ctr['Idio'] = rets
        for factor in [k for k in port_factor_ctr.keys() if k != 'Idio']:
            port_factor_ctr['Idio'] = port_factor_ctr['Idio'].sub(port_factor_ctr[factor])

        holdings_df = holdings_df.reindex(rets.index)
        holdings_df = holdings_df.shift(1)
        holdings_df = holdings_df.ffill()
        holdings_df = holdings_df.fillna(0)

        port_factor_ctr = {factor: port_factor_ctr[factor].multiply(holdings_df).sum(axis=1) for factor in port_factor_ctr.keys()}
        port_factor_ctr = pd.DataFrame(port_factor_ctr)

        return port_factor_ctr
    
    def carino_attribution(ctr_df):

        # Carino attribution for port_factor_ctr
        cum_log_return = np.log(ctr_df.sum(axis=1) + 1).cumsum()
        log_return = np.log(ctr_df.sum(axis=1) + 1)
        total_log_return = cum_log_return.iloc[-1]
        carino_factor = (log_return/total_log_return) * ((np.exp(total_log_return)-1)/(np.exp(log_return)-1))

        attr = ctr_df.multiply(carino_factor, axis=0).sum(axis=0)
        return attr
        
    def get_period_returns(self, dates=None):
        # Create quarter-end dates (with 45 day offset for reporting lag)
        if dates is None:
            dates = pd.date_range(start='2014-12-31', end='2025-03-31', freq='QE') + pd.DateOffset(days=45)
        # Filter to relevant date range
        sec_rets_df = self.sec_rets_df[(min(dates)+pd.Timedelta(days=1)):max(dates)].copy()

        date_to_quarter = pd.Series(dates, index=dates)
        quarter_end_series = date_to_quarter.reindex(sec_rets_df.index, method=None)
        sec_rets_df.loc[:, 'quarter_end'] = quarter_end_series
        sec_rets_df.loc[:, 'quarter_end'] = sec_rets_df['quarter_end'].bfill()
        # Calculate quarterly returns directly using prod


        quarter_end_returns = sec_rets_df.groupby('quarter_end').apply(
            lambda x: (1 + x.drop('quarter_end', axis=1)).prod() - 1
        )

        return quarter_end_returns
    
    def get_period_idio_returns(self, factor_scores, factor_returns, dates=None):

        if dates is None: 
            dates = pd.date_range(start='2014-12-31', end='2025-03-31', freq='QE') + pd.DateOffset(days=45)

        start_date = dates.min()
        end_date = dates.max()

        tickers = factor_scores[list(factor_scores.keys())[0]].columns

        rets = self.sec_rets_df[tickers].loc[start_date:end_date]

        cur_factor_rets = factor_returns.loc[start_date:end_date]
        cur_factor_scores = {factor: factor_scores[factor].loc[start_date:end_date] for factor in factor_scores.keys()}

        port_factor_ctr = {factor: cur_factor_scores[factor].mul(cur_factor_rets[factor], axis=0).reindex(rets.index).astype(np.float64).fillna(0) for factor in factor_scores.keys()}
        port_factor_ctr['Market'] = pd.DataFrame(cur_factor_rets['Market'].values.reshape(-1, 1).repeat(len(tickers), axis=1), 
                                   index=cur_factor_rets.index, columns=tickers).reindex(rets.index).astype(np.float64).fillna(0)
        
        port_factor_ctr['Idio'] = rets
        for factor in [k for k in port_factor_ctr.keys() if k != 'Idio']:
            port_factor_ctr['Idio'] = port_factor_ctr['Idio'].sub(port_factor_ctr[factor])
        
        idio_returns = port_factor_ctr['Idio']

        date_to_quarter = pd.Series(dates, index=dates)
        quarter_end_series = date_to_quarter.reindex(idio_returns.index, method=None)
        idio_returns.loc[:, 'quarter_end'] = quarter_end_series
        idio_returns.loc[:, 'quarter_end'] = idio_returns['quarter_end'].bfill()

        quarter_end_returns = idio_returns.groupby('quarter_end').apply(
            lambda x: (1 + x.drop('quarter_end', axis=1)).prod() - 1
        )

        return quarter_end_returns