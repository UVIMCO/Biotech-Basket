library(tidyverse)
library(rRTDA)

sec_start = as.Date("2009-12-31")
sec_end = as.Date("2025-03-31")
sec_file = "O:/AARM/Direct/Top Holdings/Research/Biotech/NW/Output/security_out.csv"
sec_mapfile = "O:/AARM/Direct/Top Holdings/Research/Biotech/NW/Output/security_mapping.csv"
sec_master = "O:/AARM/Direct/Top Holdings/Research/Biotech/NW/Output/security_master.csv"
fields = c("TOT_RETURN_INDEX_GROSS_DVDS", "PX_LAST","PX_VOLUME", "CUR_MKT_CAP")
cols_to_fill = c("TOT_RETURN_INDEX_GROSS_DVDS", "PX_LAST", "unadj_PX_LAST", "CUR_MKT_CAP")
consecutive_ct = 30

ticks_to_use = read_csv(sec_master, show_col_types = FALSE) %>%
  filter(is_biotech == TRUE, !is.na(Ticker)) %>%
  mutate(Ticker = paste0(Ticker, " Equity")) %>%
  pull(Ticker)

ticks_to_use = c(ticks_to_use,
                 "XBI US Equity",
                 "IBB US Equity") %>% unique()

# Existing Data --------------------------------------------------
sec_hist = read_csv(sec_file, show_col_types = FALSE)
sec_mapping = read_csv(sec_mapfile, show_col_types = FALSE)
sec_csv = sec_hist %>%
  utils.join_remove_dup(sec_mapping, by = c("Ticker")) %>%
  drop_na(Ticker) %>% #Remove blank Tickers
  arrange(Ticker, date)

if(nrow(sec_csv %>% na.omit()) != 0){ #If existing file has data
  sec_csv_ticks = distinct(sec_csv, Ticker)
  sec_csv$date = utils.excel_dates_as_chr_formatter(sec_csv$date)
  sec_csv$Min_Px_Date = utils.excel_dates_as_chr_formatter(sec_csv$Min_Px_Date)
  sec_csv$Max_Px_Date = utils.excel_dates_as_chr_formatter(sec_csv$Max_Px_Date)

  ## Check for existing vs. new Tickers
  exist_ticks = intersect({{ ticks_to_use }}, sec_csv_ticks$Ticker)
  new_ticks = setdiff({{ ticks_to_use }}, exist_ticks)
}else{ #If file is blank
  exist_ticks = list()
  new_ticks = {{ ticks_to_use }}
}

# New Tickers -----------------------------------------------------------
## Pull if needed
if(length(new_ticks) != 0){
  data_out_new = th_pull_sec_hist_data(ticks = new_ticks,
                                       start_date = sec_start,
                                       end_date = sec_end,
                                       fields = fields)

  data_filled_new = th_sec_data_forwardfill(data_out_new, cols_to_fill) %>% arrange(Ticker, date)
} else {
  print("No new ticker to pull.")
  data_filled_new = tibble()
}

# Existing Tickers ---------------------------------------------------------
# If has existing tickers to pull
if(length(exist_ticks) != 0){
  ## Filter out non-pricing Tickers ----------------------------------
  if(({{ sec_end }} - {{ sec_start }}) >= {{ consecutive_ct }}){
    # No price for >consecutive_ct Days in existing history
    data_no_px = sec_csv %>%
      filter(Ticker %in% exist_ticks) %>%
      group_by(Ticker) %>%
      filter(date > Max_Px_Date) %>% #Remove dates before IPO date
      mutate(
        IS_NA = is.na(PX_LAST),
        NA_Group = cumsum(!IS_NA &
                            lag(IS_NA, default = FALSE)) + 1, #Distinct N/A groups
        NA_Group = if_else(IS_NA,
                           NA_Group, NA) # Keep NA_Group IDs only for N/A values
      ) %>%
      group_by(Ticker, NA_Group) %>%
      filter(!is.na(NA_Group)) %>%
      summarize(Consecutive_NA = n()) %>%
      ungroup() %>%
      filter(Consecutive_NA > {{ consecutive_ct }}) #List of >N consec. N/As

    ticks_no_px = data_no_px %>%
      distinct(Ticker) %>%
      filter(!is.na(Ticker))
  } else{
    # No adjustment if pull period length < consecutive_ct days
    ticks_no_px = tibble()
  }

  # Filter out tickers with no existing Min and Max dates
  ticks_no_date = sec_csv %>%
    subset(is.na(Min_Px_Date) & is.na(Max_Px_Date)) %>%
    distinct(Ticker) %>%
    filter(!is.na(Ticker))

  ticks_exclude = ticks_no_px %>%
    rbind(ticks_no_date) %>%
    filter(!is.na(Ticker))

  # Update PX flag in Mapping for Tickers to exclude
  sec_csv = sec_csv %>%
    mutate(PX_NA_Flag = if_else(Ticker %in% ticks_exclude, TRUE, PX_NA_Flag))

  # Existing tickers that still have prices
  tick_update_list = sec_csv %>%
    filter(Ticker %in% exist_ticks) %>%
    distinct(Ticker) %>%
    filter(!is.na(Ticker)) %>%
    filter(!(Ticker %in% ticks_exclude))

  ticks_update = tick_update_list$Ticker

  # Dates to pull
  sec_csv_nofill = sec_csv %>%
    filter(Filled_Flag == FALSE)

  max_existing_date = max(sec_csv_nofill$date) #Max date without forward fill
  update_start = as.Date(max_existing_date)
  update_end = sec_end

  # Pull and Adjust -----------------------
  # Pull security history for new dates
  data_update = th_pull_sec_hist_data(ticks = ticks_update,
                                      start_date = update_start,
                                      end_date = update_end,
                                      fields = fields)
  # Compare latest existing date
  data_startdate = data_update %>%
    filter(date == update_start)
  csv_startdate = sec_csv %>%
    filter((date == update_start) & !(Ticker %in% ticks_exclude))

  # Use price diff to flag re-pull population
  startdate_comb = csv_startdate %>%
    bind_rows(data_startdate) %>%
    group_by(Ticker) %>%
    summarize(Px_Diff = diff(PX_LAST)) %>%
    ungroup()

  ## Tickers with Corp Actions -----------------------
  tick_repull_tbl = startdate_comb %>%
    filter(Px_Diff != 0)
  if(nrow(tick_repull_tbl %>% na.omit()) != 0){ #If need to repull
    # Re-pull entire history
    ticks_repull = tick_repull_tbl$Ticker
    sec_csv_clean = sec_csv %>%
      drop_na(Min_Px_Date) #Remove NA to not impact repull_start
    repull_start = min(sec_csv_clean$Min_Px_Date)
    repull_end = update_end
    data_repull = th_pull_sec_hist_data(ticks = ticks_repull,
                                          start_date = repull_start,
                                          end_date = repull_end,
                                          fields = fields)
    data_repull_filled = data_repull %>%
      th_sec_data_forwardfill(cols_to_fill = cols_to_fill)
  }else{
    ticks_repull = list()
  }

  ## Tickers w/o Corp Actions -----------------------
  # New data with forward fill
  data_no_repull = data_update %>%
    filter(!(Ticker %in% ticks_repull)) %>%
    th_sec_data_forwardfill(cols_to_fill = {{ cols_to_fill }}) %>%
    arrange(Ticker, date) %>%
    # Calculate TR daily change % and remove TR index level
  mutate(TR_Daily_Chg = TOT_RETURN_INDEX_GROSS_DVDS /
           lag(TOT_RETURN_INDEX_GROSS_DVDS) - 1) %>%
    mutate(TOT_RETURN_INDEX_GROSS_DVDS = NA) %>%
    filter(date > update_start) #Remove overlapping date

    # Existing data
  csv_no_repull = sec_csv %>%
    filter(!(Ticker %in% ticks_repull))

    # Combine new and old data history
  no_repull_comb = csv_no_repull %>%
    bind_rows(data_no_repull) %>%
    arrange(Ticker, date)

    # Recalculate Total Return Level for new data set
    no_repull_recalc = no_repull_comb %>%
      group_by(Ticker) %>%
      mutate(
        TR_Cumulative_Chg = if_else(is.na(TR_Daily_Chg),
                                    NA_real_,
                                    cumprod(replace_na((TR_Daily_Chg + 1), 1))),
        #Replace NA with 1 to continue cumulative calc
        TR_New = if_else(
          (is.na(TOT_RETURN_INDEX_GROSS_DVDS) == TRUE) & (is.na(TR_Daily_Chg) == FALSE),
          na.locf(TOT_RETURN_INDEX_GROSS_DVDS, na.rm = FALSE, fromLast = FALSE) * TR_Cumulative_Chg,
          TOT_RETURN_INDEX_GROSS_DVDS)
      ) %>%
      ungroup() %>%
      select(-c(TOT_RETURN_INDEX_GROSS_DVDS, TR_Cumulative_Chg, TR_Daily_Chg)) %>%
      rename(TOT_RETURN_INDEX_GROSS_DVDS = TR_New) %>%
      arrange(Ticker, date)

    ## Combined Output
    if(nrow(tick_repull_tbl %>% na.omit()) != 0){ #If need to repull
      data_update_out = no_repull_recalc %>%
        select(all_of(names(data_repull_filled))) %>% #Reorder columns
        bind_rows(data_repull_filled)

      # Update min and max date to reflect entire history
      data_update_out = data_update_out %>%
        group_by(Ticker) %>%
        mutate(
          Min_Px_Date = min(Min_Px_Date),
          Max_Px_Date = max(Max_Px_Date)
        ) %>%
        ungroup()
    }else{
      data_update_out = no_repull_recalc
    }

    # Forward fill handling
    data_update_out = data_update_out %>%
      utils.join_remove_dup(data_update_out, by = c("Ticker", "date"))

    data_filled = data_update_out %>%
      select(-c("Min_Px_Date", "Max_Px_Date")) %>%
      th_sec_data_forwardfill(cols_to_fill = cols_to_fill) %>%
      arrange(Ticker, date)

    # Output -----------------------------------------------
    # Reorder columns for new ticker tibble if needed
    if(length(new_ticks) != 0){
      data_filled_new_reorder = data_filled_new %>%
        select(one_of(names(data_filled)))
    }else{
      data_filled_new_reorder = tibble()
    }
    data_csv = data_filled %>%
      rbind(data_filled_new_reorder) %>%
      drop_na(Ticker) %>%
      relocate(TOT_RETURN_INDEX_GROSS_DVDS, .after = date) %>%
      relocate(PX_NA_Flag, .before =  Min_Px_Date) %>%
      relocate(Filled_Flag, .after = last_col()) %>%
      arrange(Ticker, date)

}else{ #No existing ticker to pull
  print("No existing ticker to pull.")
  data_csv = data_filled_new %>%
    select(all_of(names(sec_csv))) %>% #Reorder columns
    bind_rows(sec_csv) %>%
    drop_na(Ticker) %>%
    arrange(Ticker, date)
}

# Export Security Data -----------------------------------------------
# Export entire history
# Use flags as of latest date
sec_mapping_new = data_csv %>%
  select(Ticker, date, PX_NA_Flag, Min_Px_Date, Max_Px_Date) %>%
  arrange(Ticker, date) %>%
  group_by(Ticker) %>%
  slice_max(order_by = date, n = 1) %>%
  ungroup() %>%
  distinct(Ticker, PX_NA_Flag, Min_Px_Date, Max_Px_Date)

# Data without flag columns
data_out = data_csv %>%
  select(-c(PX_NA_Flag, Min_Px_Date, Max_Px_Date)) %>%
  arrange(Ticker, date)

# Export separately
write_csv(sec_mapping_new, sec_mapfile)
write_csv(data_out, sec_file)

print("Security Data Write Done")

