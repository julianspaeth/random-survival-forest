from random_survival_forest import RandomSurvivalForest
from random_survival_forest.scoring import concordance_index
from sklearn.model_selection import train_test_split
import pandas as pd

print("Load Dataset")
x_train = pd.read_csv("/home/spaethju/Dokumente/master-thesis/data/icgc/blood/large/features/final/cnsm_features.csv")
x_train = x_train.set_index("icgc_donor_id")
x_train = x_train.astype("int16")
y_train = pd.read_csv("/home/spaethju/Dokumente/master-thesis/data/icgc/blood/large/features/final/cnsm_labels.csv").set_index("icgc_donor_id")
y_train = y_train[["donor_survival_time", "donor_vital_status"]].replace({"alive": 0, "deceased": 1})
y_train["donor_survival_time"] = y_train["donor_survival_time"].apply(lambda x: round(x, 1))
y_train["donor_survival_time"] = y_train["donor_survival_time"].astype("float16")
y_train["donor_vital_status"] = y_train["donor_vital_status"].astype("int8")
timeline = range(0, 15, 1)

x_train = x_train.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, shuffle=True,
                                                  test_size=0.33, random_state=10)
x_train = x_train.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
x_val = x_val.reset_index(drop=True)
y_val = y_val.reset_index(drop=True)

rsf = RandomSurvivalForest(n_estimators=50, timeline=timeline, min_leaf=1, unique_deaths=1, n_jobs=-1, random_state=1)
rsf.fit(x_train, y_train)

oob_c = round(rsf.compute_oob_score(), 3)

y_pred = rsf.predict(x_val)
val_c = round(concordance_index(y_time=y_val["donor_survival_time"], y_pred=y_pred,
                                y_event=y_val["donor_vital_status"]), 3)

print("OOB:", oob_c,  "Test:", val_c)
