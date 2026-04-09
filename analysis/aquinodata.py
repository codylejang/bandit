
# from experiment_config import AQUINO_BEH_PQT_DATA



def load_aquino_pqt(): 
    import polars as pl
    from analysis.experiment_config import AQUINO_BEH_PQT_DATA    
    DF = pl.read_parquet(AQUINO_BEH_PQT_DATA)
    DF = DF[
        [
        "patientId","trialAccum","blockID","trialInBlock","outcome",
        "selectedVector",
        "select_qVals","reject_qVals","qValsLeft","qValsRight",
        "select_uVal","reject_uVal","uValLeft", "uValRight",             
        "select_nVal","reject_nVal", "nValLeft", "nValRight",
        "Q1Left", "Q1Right", "Q2Left", "Q2Right",
        "Q3Left", "Q3Right", "Q4Left", "Q4Right",
        "Q5Left", "Q5Right"
        ]
    ]
    print(DF)
    # DF = DF.filter(pl.col("trialInBlock")<=15)
    return DF

# PREP DATA
def init_aquino_df():
    import polars as pl

    DF = load_aquino_pqt()

    DF = (DF.with_columns(pl.col("patientId").cast(pl.Int64))
        .sort(["patientId","blockID"]))
        
    # compute differences
    DF = DF.with_columns([
        (pl.col("selectedVector")==1).cast(pl.Int8).alias("chose_left"),    
        (pl.col("qValsLeft")-pl.col("qValsRight")).alias("dQ"),
        (pl.col("uValLeft")-pl.col("uValRight")).alias("dUnc"),
        (pl.col("nValLeft")-pl.col("nValRight")).alias("dNov"),

        (pl.col("select_qVals")-pl.col("reject_qVals")).alias("dQ_select"),

        pl.when(pl.col("select_qVals") > pl.col("reject_qVals")).then(1.0)
        .when(pl.col("select_qVals") < pl.col("reject_qVals")).then(0.0)
        .otherwise(None)
        .alias("chose_ev"),

        pl.when(pl.col("select_uVal") > pl.col("reject_uVal")).then(1.0)
        .when(pl.col("select_uVal") < pl.col("reject_uVal")).then(0.0)
        .otherwise(None)
        .alias("chose_uncertain"),

        pl.when(pl.col("select_nVal") > pl.col("reject_nVal")).then(1.0)
        .when(pl.col("select_nVal") < pl.col("reject_nVal")).then(0.0)
        .otherwise(None)
        .alias("chose_novel"),

    ])    

    DF = DF.sort(["patientId","blockID"])
    print(DF)


    n_quantiles = 5;
    DF = DF.group_by("patientId").map_groups(
        lambda g: g.with_columns([
            pl.col("dQ").qcut(n_quantiles, allow_duplicates=True).rank("dense").cast(pl.Int8).alias("dQ_binned"),
            pl.col("dUnc").qcut(n_quantiles, allow_duplicates=True).rank("dense").cast(pl.Int8).alias("dUnc_binned"),
            pl.col("dNov").qcut(n_quantiles, allow_duplicates=True).rank("dense").cast(pl.Int8).alias("dNov_binned"),
            pl.col("dQ_select").qcut(n_quantiles, allow_duplicates=True).rank("dense").cast(pl.Int8).alias("dQ_select_binned")
        ])
    )
    
    return DF


# VALIDATE
def validate_aquino_data(DF):
    tol = 1e-9

    DF_check = DF.with_columns([
        (
            ((pl.col("select_qVals") - pl.col("qValsLeft")).abs() < tol) &
            ((pl.col("select_uVal") - pl.col("uValLeft")).abs() < tol) &
            ((pl.col("select_nVal") - pl.col("nValLeft")).abs() < tol)
        ).alias("selected_matches_left"),

        (
            ((pl.col("select_qVals") - pl.col("qValsRight")).abs() < tol) &
            ((pl.col("select_uVal") - pl.col("uValRight")).abs() < tol) &
            ((pl.col("select_nVal") - pl.col("nValRight")).abs() < tol)
        ).alias("selected_matches_right"),
    ])

    print(
        DF_check.select([
            (((pl.col("selectedVector") == 1) == pl.col("selected_matches_left")).mean()).alias("agreement_left"),
            (((pl.col("selectedVector") != 1) == pl.col("selected_matches_right")).mean()).alias("agreement_right"),
        ])
    )

