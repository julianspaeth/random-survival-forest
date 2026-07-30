import time

from lifelines import datasets
from sklearn.model_selection import train_test_split

from random_survival_forest.models import RandomSurvivalForest
from random_survival_forest.scoring import concordance_index

# Rossi recidivism dataset: "arrest" is the event indicator, "week" is the time to event/censoring.
rossi = datasets.load_rossi()
# Attention: event column must be index 0, duration column index 1 in y
y = rossi.loc[:, ["arrest", "week"]]
X = rossi.drop(["arrest", "week"], axis=1)
X, X_test, y, y_test = train_test_split(X, y, test_size=0.33, random_state=10)

print("Start training...")
start_time = time.time()
rsf = RandomSurvivalForest(n_estimators=50, n_jobs=-1, random_state=42)
rsf = rsf.fit(X, y)
print(f'--- {round(time.time() - start_time, 3)} seconds ---')

# Predicted cumulative hazard function per test sample.
y_pred = rsf.predict(X_test)

# Concordance index: fraction of comparable sample pairs correctly ranked by predicted risk.
c_val = concordance_index(y_time=y_test["week"], y_pred=y_pred, y_event=y_test["arrest"])
print(f'C-index {round(c_val, 3)}')
