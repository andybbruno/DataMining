from imblearn.under_sampling import RandomUnderSampler

df = pd.read_csv(file)

y = df['IsBadBuy']
X = df.drop(columns=['IsBadBuy'])
X, y = RandomUnderSampler().fit_resample(X, y)