def anz_fig_1F(DF):
    # Assumes the following columns are present in data frame:
    # - "patientId","trialInBlock", "dQ", "dUnc", "dNov", "dQ","dUnc","dNov","chose_left"

    from analysis.experiment_config import BANDIT_FEATURES
    from analysis.helpers import norm_list
    import polars as pl
    import numpy as np
    import statsmodels.api as sm
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import statsmodels.api as sm
    # from scipy 


    target_col = "chose_left"
    df = DF.select(norm_list(["patientId","blockID", "trialInBlock", *BANDIT_FEATURES, target_col]))
    feature_cols = ["patientId","trialInBlock", "dQ", "dUnc", "dNov"]

    df = (
        df
        .select([*feature_cols, target_col])
        .drop_nulls()
        .with_columns([
            (pl.col("trialInBlock") * pl.col("dQ")).alias("trial_dQ"),
            (pl.col("trialInBlock") * pl.col("dUnc")).alias("trial_dUnc"),
            (pl.col("trialInBlock") * pl.col("dNov")).alias("trial_dNov"),
        ])
    )

    feature_cols_use = ["dQ", "dUnc", "dNov", "trial_dQ", "trial_dUnc", "trial_dNov"]
    df_coeffs = [] 

    for ii in df["patientId"].unique().to_list():
        print(ii)
        df_patient = df.filter(pl.col("patientId")==ii)
    
        X = sm.add_constant(df_patient.select(feature_cols_use).to_pandas(), has_constant="add")
        y = df_patient.select(target_col).to_series().to_numpy()
        logit_mod = sm.Logit(y, X).fit(disp=False)

        patient_coef = pd.DataFrame({
            "patientId": ii,
            "term": logit_mod.params.index,
            "coef": logit_mod.params,
            "std_err": logit_mod.bse,
            "z": logit_mod.tvalues,
            "p_value": logit_mod.pvalues,
            "odds_ratio": np.exp(logit_mod.params)
        }).round(4)

        df_coeffs.append(patient_coef)


    print(df_coeffs)
    df_plot = pd.concat(df_coeffs, ignore_index=True)  # make a DataFrame
    df_plot = df_plot.loc[df_plot["term"] != "const"].copy()  # drop intercept
    plt.figure(figsize=(8, 4))
    sns.violinplot(data=df_plot, x="term", y="coef", inner="box", cut=0)
    sns.stripplot(data=df_plot, x="term", y="coef", color="k", alpha=0.25, jitter=0.15, size=3)
    plt.axhline(0, color="gray", lw=1, ls="--")
    plt.ylabel("Logit coefficient")
    plt.xlabel("")
    plt.tight_layout()
    plt.show()

 
    return patient_coef