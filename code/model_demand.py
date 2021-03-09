def setup_sim(df_res, as_of_date="2017-08-01"):
    """
    Takes reservations and returns a DataFrame that can be used as a revenue management simulation.

    Very similar to setup.df_to_dbd (does the same thing but uses predicted cancels instead of actual)

    Our data is made up of reservations containing 'Arrival Date' and 'Length of Stay'.
    This function is used to determine how many rooms were sold on a given night, accounting for
    guests that arrived previously and are staying multiple nights.

    ____
    Parameters:
        - df_res (pandas.DataFrame, required): reservations DataFrame containing "will_cancel" column
        - as_of_date (str ("%Y-%m-%d"), optional): resulting day-by-days DataFrame will start on this day
        - cxl_type (str, optional): either "a" (actual) or "p" (predicted). Default value is "p".
    """
    df_dates = df_res.copy()
    date = pd.to_datetime(first_date, format="%Y-%m-%d")
    end_date = datetime.date(2017, 8, 31)
    delta = datetime.timedelta(days=1)
    max_los = int(df_dates["LOS"].max())

    nightly_stats = {}

    while date <= end_date:

        date_string = datetime.datetime.strftime(date, format="%Y-%m-%d")
        tminus = 0

        # initialize date dict, which will go into nightly_stats as {'date': {'stat': 'val', 'stat', 'val'}}
        date_stats = defaultdict(int)

        # start on the arrival date and move back
        # to capture ALL reservations touching 'date' (and not just those that arrive on 'date')
        for _ in range(max_los):

            #
            date_tminus = date - pd.DateOffset(tminus)

            date_tminus_string = datetime.datetime.strftime(
                date_tminus, format="%Y-%m-%d"
            )

            mask = (
                (df_dates.ArrivalDate == date_tminus_string)
                & (df_dates.LOS >= 1 + tminus)
                & (df_dates.IsCanceled == 0)
            )

            date_stats["RoomsOTB"] += len(df_dates[mask])
            date_stats["RevOTB"] += df_dates[mask].ADR.sum()

            tmp = (
                df_dates[mask][["ResNum", "CustomerType", "ADR"]]
                .groupby("CustomerType")
                .agg({"ResNum": "count", "ADR": "sum"})
                .rename(columns={"ResNum": "RS", "ADR": "Rev"})
            )

            if "Transient" in list(tmp.index):
                date_stats["Trn_RoomsOTB"] += tmp.loc["Transient", "RS"]
                date_stats["Trn_RevOTB"] += tmp.loc["Transient", "Rev"]

            if "Transient-Party" in list(tmp.index):
                date_stats["TrnP_RoomsOTB"] += tmp.loc["Transient-Party", "RS"]
                date_stats["TrnP_RevOTB"] += tmp.loc["Transient-Party", "Rev"]

            if "Group" in list(tmp.index):
                date_stats["Grp_RoomsOTB"] += tmp.loc["Group", "RS"]
                date_stats["Grp_RevOTB"] += tmp.loc["Group", "Rev"]

            if "Contract" in list(tmp.index):
                date_stats["Cnt_RoomsOTB"] += tmp.loc["Contract", "RS"]
                date_stats["Cnt_RevOTB"] += tmp.loc["Contract", "Rev"]

            tminus += 1

        nightly_stats[date_string] = dict(date_stats)
        date += delta

    return pd.DataFrame(nightly_stats).transpose()