from imblearn.under_sampling import RandomUnderSampler
import pandas as pd

file = "kids_train_cleaned.csv"
df = pd.read_csv(file)
df['WheelType'].fillna('UNKNOWN', inplace=True)

y = df['IsBadBuy']
X = df.drop(columns=['IsBadBuy'])

cols = X.columns
X, y = RandomUnderSampler().fit_resample(X, y)


new = pd.DataFrame(data=X, columns=cols)
new_y = pd.DataFrame(data=y, columns=['IsBadBuy'])
new['IsBadBuy'] = new_y
new.to_csv('5050_kids_train_cleaned.csv')
