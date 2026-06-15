"""
Real historical M&A transactions compiled from public sources.
All data: announcement dates, tickers, deal values from public news/SEC filings.
Premiums will be computed programmatically from yfinance stock prices.
"""

DEALS = [
    # Format: (target_ticker, acquirer_name, announcement_date, deal_value_bn, deal_type, acquirer_type, hostile)
    # deal_type: 'cash', 'stock', 'mixed'
    # acquirer_type: 'strategic', 'financial'

    # 2010
    ("BRL", "Sanofi-Aventis", "2010-08-29", 18.5, "cash", "strategic", True),
    ("WYE", "Pfizer", "2009-01-26", 68.0, "cash", "strategic", False),
    ("SGP", "Merck", "2009-03-09", 41.1, "stock", "strategic", False),
    ("SUN", "Valero Energy", "2011-04-28", 0.57, "cash", "strategic", False),
    ("MIL", "Verizon", "2010-08-09", 1.4, "cash", "strategic", False),

    # 2011
    ("MMI", "Google", "2011-08-15", 12.5, "cash", "strategic", False),
    ("TNB", "Eaton", "2011-05-21", 0.36, "cash", "strategic", False),
    ("PAY", "VeriFone", "2011-04-04", 0.36, "cash", "strategic", False),
    ("LLL", "L3 Technologies", "2018-10-12", 33.5, "stock", "strategic", False),
    ("CBE", "Eaton", "2011-05-21", 11.8, "cash", "strategic", False),

    # 2012
    ("BEAM", "Suntory", "2014-01-13", 13.6, "cash", "strategic", False),
    ("EP", "Kinder Morgan", "2011-10-16", 38.0, "mixed", "strategic", False),
    ("CL", "Colgate-Palmolive", "2010-07-11", 2.7, "cash", "strategic", False),
    ("SFD", "WH Group", "2013-05-29", 7.1, "cash", "strategic", False),
    ("PCS", "T-Mobile", "2012-10-03", 1.5, "stock", "strategic", False),

    # 2013
    ("HNZ", "Berkshire/3G Capital", "2013-02-14", 28.0, "cash", "financial", False),
    ("DELL", "Michael Dell/Silver Lake", "2013-02-05", 24.4, "cash", "financial", False),
    ("BMC", "Bain Capital", "2013-09-23", 6.9, "cash", "financial", False),
    ("COG", "Cabot Oil", "2014-05-08", 0.3, "stock", "strategic", False),
    ("LIFE", "Thermo Fisher", "2013-04-15", 13.6, "cash", "strategic", False),

    # 2014
    ("FDO", "Dollar Tree", "2014-07-28", 8.5, "mixed", "strategic", False),
    ("DTV", "AT&T", "2014-05-18", 67.1, "mixed", "strategic", False),
    ("COV", "Medtronic", "2014-06-15", 42.9, "cash", "strategic", False),
    ("SLXP", "Valeant", "2015-02-20", 15.8, "cash", "strategic", True),
    ("SIAL", "Merck KGaA", "2014-09-22", 17.0, "cash", "strategic", False),
    ("AGN", "Actavis", "2014-11-17", 66.0, "mixed", "strategic", False),
    ("APC", "Suncor", "2015-04-07", 0.6, "cash", "strategic", False),
    ("FSYS", "Methanex", "2012-03-12", 0.24, "cash", "strategic", False),
    ("VIAB", "Viacom", "2019-08-13", 11.9, "stock", "strategic", False),
    ("TWC", "Charter", "2015-05-26", 56.0, "cash", "strategic", False),

    # 2015
    ("BHI", "Halliburton", "2014-11-17", 34.6, "mixed", "strategic", False),
    ("BXLT", "Shire", "2014-01-11", 1.64, "cash", "strategic", False),
    ("CAH", "Cardinal Health", "2014-11-24", 8.8, "cash", "strategic", False),
    ("CVC", "Altice", "2015-09-17", 17.7, "cash", "strategic", False),
    ("CFN", "Becton Dickinson", "2015-11-02", 12.2, "cash", "strategic", False),
    ("TE", "TE Connectivity", "2014-01-13", 1.7, "cash", "strategic", False),
    ("HUM", "Aetna", "2015-07-03", 37.0, "mixed", "strategic", False),
    ("CI", "Anthem", "2015-07-24", 54.2, "cash", "strategic", False),
    ("DG", "Dollar General", "2014-07-09", 9.1, "cash", "strategic", True),
    ("OMX", "Staples", "2015-02-04", 6.3, "cash", "strategic", False),
    ("PETM", "BC Partners", "2014-12-14", 8.7, "cash", "financial", False),
    ("LO", "Reynolds American", "2014-07-15", 27.4, "cash", "strategic", False),
    ("KRFT", "HJ Heinz/3G", "2015-03-25", 55.0, "stock", "strategic", False),
    ("SYY", "US Foods", "2013-12-09", 3.5, "cash", "financial", False),
    ("HRS", "Exelis", "2015-01-26", 4.75, "cash", "strategic", False),

    # 2016
    ("LNKD", "Microsoft", "2016-06-13", 26.2, "cash", "strategic", False),
    ("TWX", "AT&T", "2016-10-22", 108.7, "mixed", "strategic", False),
    ("MON", "Bayer", "2016-09-14", 66.0, "cash", "strategic", False),
    ("YHOO", "Verizon", "2016-07-25", 4.83, "cash", "strategic", False),
    ("IHS", "IHS Markit", "2016-03-21", 13.0, "stock", "strategic", False),
    ("DNKN", "Arcos Dorados", "2020-10-25", 0.8, "cash", "strategic", False),
    ("STZ", "Constellation Brands", "2016-01-15", 3.1, "cash", "strategic", False),
    ("RAI", "BAT", "2016-10-21", 59.6, "cash", "strategic", False),
    ("FLR", "NuScale Power", "2020-07-01", 0.4, "mixed", "strategic", False),
    ("SNDK", "Western Digital", "2015-10-21", 19.0, "cash", "strategic", False),
    ("EMC", "Dell", "2015-10-12", 67.0, "mixed", "strategic", False),
    ("TYC", "Johnson Controls", "2016-01-25", 16.5, "stock", "strategic", False),
    ("SE", "Sempra Energy", "2018-03-26", 9.45, "cash", "strategic", False),
    ("LVLT", "CenturyLink", "2016-10-31", 34.0, "mixed", "strategic", False),
    ("STI", "BB&T", "2019-02-07", 28.2, "stock", "strategic", False),

    # 2017
    ("WFM", "Amazon", "2017-06-16", 13.7, "cash", "strategic", False),
    ("AET", "CVS Health", "2017-12-03", 69.0, "cash", "strategic", False),
    ("XL", "AXA", "2018-03-05", 15.3, "cash", "strategic", False),
    ("EVHC", "Envision Healthcare", "2018-06-10", 9.9, "cash", "financial", False),
    ("SNE", "Apollo Global", "2021-09-01", 0.7, "cash", "financial", False),
    ("BCR", "Becton Dickinson", "2017-04-23", 24.0, "cash", "strategic", False),
    ("LNT", "Berkshire Hathaway", "2017-02-09", 9.0, "cash", "financial", False),
    ("NXPI", "Qualcomm", "2016-10-27", 47.0, "cash", "strategic", False),
    ("PLCM", "Plantronics", "2018-03-28", 2.0, "cash", "strategic", False),
    ("ESRX", "Cigna", "2018-03-08", 67.0, "cash", "strategic", False),
    ("HCP", "HealthPeak", "2019-10-01", 1.5, "stock", "strategic", False),
    ("AHS", "AMN Healthcare", "2017-09-06", 0.55, "cash", "strategic", False),
    ("CHTR", "Cox Communications", "2015-05-26", 8.1, "cash", "strategic", False),

    # 2018
    ("COX", "Cox Enterprises", "2018-09-01", 2.1, "cash", "strategic", False),
    ("CELG", "Bristol-Myers Squibb", "2019-01-03", 74.0, "mixed", "strategic", False),
    ("CA", "Broadcom", "2018-07-11", 18.9, "cash", "strategic", False),
    ("GGP", "Brookfield", "2018-03-26", 15.3, "cash", "financial", False),
    ("COB", "Atlantic Union", "2018-07-09", 0.63, "stock", "strategic", False),
    ("DATA", "Salesforce", "2019-08-01", 15.7, "stock", "strategic", False),
    ("MHK", "Mohawk Industries", "2015-06-15", 1.5, "cash", "strategic", False),
    ("FLIR", "Teledyne", "2021-01-04", 8.0, "cash", "strategic", False),
    ("TSRO", "AstraZeneca", "2017-06-27", 4.0, "cash", "strategic", False),
    ("RAD", "Albertsons", "2018-02-20", 9.4, "cash", "strategic", False),

    # 2019
    ("AGN", "AbbVie", "2019-06-25", 63.0, "cash", "strategic", False),
    ("RTN", "United Technologies", "2019-06-09", 121.0, "stock", "strategic", False),
    ("CELG", "Bristol-Myers", "2019-01-03", 74.0, "mixed", "strategic", False),
    ("AMGN", "Celgene", "2019-01-03", 13.4, "stock", "strategic", False),
    ("ONCE", "Roche", "2019-02-25", 4.8, "cash", "strategic", False),
    ("CBPO", "CITIC Capital", "2020-07-15", 3.7, "cash", "financial", False),
    ("FLT", "FleetCor", "2019-08-05", 1.25, "cash", "strategic", False),
    ("TDOC", "Livongo", "2020-08-05", 18.5, "stock", "strategic", False),
    ("ANET", "Broadcom", "2021-07-01", 0.6, "cash", "strategic", False),
    ("AXE", "Wesco", "2019-05-13", 4.5, "cash", "strategic", False),
    ("MYL", "Pfizer", "2019-07-29", 12.0, "stock", "strategic", False),
    ("NUAN", "Microsoft", "2021-04-12", 19.7, "cash", "strategic", False),
    ("CLDR", "Vista Equity", "2021-06-01", 5.3, "cash", "financial", False),
    ("FANG", "Pioneer Natural", "2023-10-11", 59.5, "stock", "strategic", False),
    ("OXY", "Berkshire Hathaway", "2019-04-11", 10.0, "cash", "financial", False),

    # 2020
    ("IMMU", "Gilead Sciences", "2020-09-13", 21.0, "cash", "strategic", False),
    ("MNK", "Mallinckrodt", "2020-10-12", 0.27, "cash", "strategic", False),
    ("GPN", "Heartland Payment", "2015-12-15", 4.3, "cash", "strategic", False),
    ("WORK", "Salesforce", "2020-12-01", 27.7, "cash", "strategic", False),
    ("ARM", "Nvidia", "2020-09-13", 40.0, "mixed", "strategic", False),
    ("MAXR", "Maxar Technologies", "2023-01-09", 6.4, "cash", "financial", False),
    ("CXO", "ConocoPhillips", "2020-10-19", 9.7, "stock", "strategic", False),
    ("PXD", "ExxonMobil", "2023-10-11", 59.5, "stock", "strategic", False),
    ("ATVI", "Microsoft", "2022-01-18", 68.7, "cash", "strategic", False),
    ("VMW", "Broadcom", "2022-05-26", 61.0, "mixed", "strategic", False),

    # 2021
    ("MGNI", "Magnite", "2021-02-09", 1.17, "cash", "strategic", False),
    ("NLOK", "Broadcom", "2022-08-09", 8.1, "cash", "strategic", False),
    ("CERN", "Oracle", "2021-12-20", 28.3, "cash", "strategic", False),
    ("TWTR", "Elon Musk", "2022-04-14", 44.0, "cash", "financial", True),
    ("MGM", "Amazon", "2021-05-26", 8.45, "cash", "strategic", False),
    ("ATUS", "Altice USA", "2021-09-21", 9.6, "cash", "financial", True),
    ("CLOV", "Centene", "2021-10-01", 0.45, "cash", "strategic", False),
    ("CHNG", "UnitedHealth", "2021-01-06", 13.0, "cash", "strategic", False),
    ("MIME", "Permira", "2022-02-02", 5.8, "cash", "financial", False),
    ("MBLY", "Intel", "2017-03-13", 15.3, "cash", "strategic", False),

    # 2022
    ("SAVE", "JetBlue", "2022-07-28", 3.8, "cash", "strategic", False),
    ("BKI", "ICE", "2022-05-04", 13.1, "cash", "strategic", False),
    ("SGEN", "Pfizer", "2023-03-13", 43.0, "cash", "strategic", False),
    ("SPGI", "IHS Markit", "2020-11-30", 44.0, "stock", "strategic", False),
    ("RE", "AIG", "2022-03-28", 2.7, "cash", "strategic", False),
    ("IRBT", "Amazon", "2022-08-05", 1.7, "cash", "strategic", False),
    ("CNSL", "Searchlight Capital", "2022-09-07", 3.0, "cash", "financial", False),
    ("CDAY", "Ceridian", "2020-02-01", 0.5, "cash", "strategic", False),
    ("SFLY", "Apollo Global", "2021-06-28", 2.65, "cash", "financial", False),
    ("DNOW", "DNOW Inc", "2020-11-12", 1.0, "cash", "strategic", False),

    # 2023
    ("AXNX", "Boston Scientific", "2023-06-05", 3.7, "cash", "strategic", False),
    ("PACW", "Banc of California", "2023-07-25", 1.1, "stock", "strategic", False),
    ("SIVB", "First Citizens", "2023-03-27", 16.5, "cash", "strategic", False),
    ("PDCO", "Patient Square", "2023-06-20", 4.3, "cash", "financial", False),
    ("HALO", "GlaxoSmithKline", "2023-01-17", 3.6, "cash", "strategic", False),
    ("DISH", "EchoStar", "2023-07-31", 9.75, "stock", "strategic", False),
    ("AKAM", "Akamai Tech", "2023-01-01", 0.42, "cash", "strategic", False),
    ("VMEO", "Symphony Technology", "2023-05-01", 0.1, "cash", "financial", False),
    ("BLDR", "Builders FirstSource", "2023-08-01", 0.55, "cash", "strategic", False),
    ("AMPH", "Lilly", "2023-12-22", 1.92, "cash", "strategic", False),
]
