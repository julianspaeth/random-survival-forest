from random_survival_forest import RandomSurvivalForest
from random_survival_forest.scoring import concordance_index
from sklearn.model_selection import train_test_split, KFold
from lifelines import AalenAdditiveFitter, CoxPHFitter, WeibullAFTFitter, LogNormalAFTFitter, LogLogisticAFTFitter
import pandas as pd
import numpy as np
from lifelines.utils import concordance_index

print("Load SSM Dataset")
x_train = pd.read_csv("/home/spaethju/Dokumente/master-thesis/data/icgc/blood/large/features/final/ssm_features.csv")
x_train = x_train.set_index("icgc_donor_id")
x_train = x_train.astype("int16")
y_train = pd.read_csv("/home/spaethju/Dokumente/master-thesis/data/icgc/blood/large/features/final/ssm_labels.csv").\
    set_index("icgc_donor_id")
y_train = y_train[["donor_survival_time", "donor_vital_status"]].replace({"alive": 0, "deceased": 1})
y_train["donor_survival_time"] = y_train["donor_survival_time"].apply(lambda x: round(x, 1))
y_train["donor_survival_time"] = y_train["donor_survival_time"].astype("float16")
y_train["donor_vital_status"] = y_train["donor_vital_status"].astype("int8")
timeline = range(0, 10, 1)

x_train = x_train.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, shuffle=True,
                                                  test_size=0.33, random_state=10)
x_train = x_train.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
x_val = x_val.reset_index(drop=True)
y_val = y_val.reset_index(drop=True)

x = x_train.copy()
x["T"] = y_train["donor_survival_time"]
x["E"] = y_train["donor_vital_status"]

fitters = [WeibullAFTFitter(), AalenAdditiveFitter(coef_penalizer=0.1),
           LogNormalAFTFitter(), LogLogisticAFTFitter()]


def drop_correlated_features(x, thresh=0.95):
    x_drop_correlated = x.copy()
    # Create correlation matrix
    corr_matrix = x_drop_correlated.corr().abs()
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    # Find index of feature columns with correlation greater than 0.95
    to_drop = [column for column in upper.columns if any(upper[column] > thresh)]
    x_drop_correlated = x_drop_correlated.drop(x_drop_correlated[to_drop], axis=1)

    return x_drop_correlated, to_drop


train_c = {}
val_c = {}
for fitter in fitters:
    train_c[fitter] = []
    val_c[fitter] = []
train_c["CPH"] = []
val_c["CPH"] = []

kfold = KFold(n_splits=5, random_state=10, shuffle=True)

for train_index, test_index in kfold.split(x):
    X_train, X_val = x.iloc[train_index, :], x.iloc[test_index, :]
    X_train_not_negative = X_train.copy()
    X_train_not_negative["T"] = X_train["T"].replace({0: 0.000001})
    for fitter in fitters:
        if isinstance(fitter, WeibullAFTFitter) or \
                isinstance(fitter, LogNormalAFTFitter) or \
                isinstance(fitter, LogLogisticAFTFitter):
            fitter.fit(X_train_not_negative, duration_col="T", event_col="E")
        else:
            fitter.fit(X_train, duration_col="T", event_col="E")
        train_c[fitter].append(fitter.score_)
        val_c[fitter].append(concordance_index(X_val["T"], -fitter.predict_median(X_val), X_val["E"]))

    cph = CoxPHFitter()
    x_drop_correlated, to_drop = drop_correlated_features(X_train.drop(["T", "E"], axis=1), 0.9)
    x_drop_correlated["T"] = X_train["T"]
    x_drop_correlated["E"] = X_train["E"]
    cph.fit(x_drop_correlated, "T", event_col="E", step_size=0.2)
    train_c["CPH"].append(cph.score_)
    x_val_drop_correlated = X_val.copy().drop(X_val[to_drop], axis=1)
    val_c["CPH"].append(concordance_index(x_val_drop_correlated["T"],
                                          -cph.predict_partial_hazard(x_val_drop_correlated),
                                          x_val_drop_correlated["E"]))

for fitter in fitters:
    print(fitter)
    print("Train:", round(np.mean(train_c[fitter]), 3))
    print("Val", round(np.mean(val_c[fitter]), 3))

print(cph)
print("Train:", round(np.mean(train_c["CPH"]), 3))
print("Val", round(np.mean(val_c["CPH"]), 3))

# rsf = RandomSurvivalForest(n_estimators=20, timeline=timeline, min_leaf=3, unique_deaths=3, n_jobs=-1, random_state=1)
# rsf.fit(x_train, y_train)

# oob_c = round(rsf.compute_oob_score(), 3)

# y_pred = rsf.predict(x_val)
# val_c = round(concordance_index(y_time=y_val["donor_survival_time"], y_pred=y_pred,
#                                 y_event=y_val["donor_vital_status"]), 3)

# print("OOB:", oob_c,  "Test:", val_c)



