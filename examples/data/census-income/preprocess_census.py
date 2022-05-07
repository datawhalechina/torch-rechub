import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

col_names = [
    'age', 'class of worker', 'industry code', 'occupation code', 'education', 'wage per hour', 'enrolled in edu inst last wk', 'marital status', 'major industry code', 'major occupation code', 'race', 'hispanic origin', 'sex', 'member of a labor union',
    'reason for unemployment', 'full or part time employment stat', 'capital gains', 'capital losses', 'divdends from stocks', 'tax filer status', 'region of previous residence', 'state of previous residence', 'detailed household and family stat',
    'detailed household summary in household', 'instance weight', 'migration code-change in msa', 'migration code-change in reg', 'migration code-move within reg', 'live in this house 1 year ago', 'migration prev res in sunbelt',
    'num persons worked for employer', 'family members under 18', 'country of birth father', 'country of birth mother', 'country of birth self', 'citizenship', 'own business or self employed', 'fill inc questionnaire for veterans admin', 'veterans benefits',
    'weeks worked in year', 'year', 'income'
]

train_data = pd.read_csv("./census-income.data", header=None, names=col_names)
test_data = pd.read_csv("./census-income.test", header=None, names=col_names)
del train_data['instance weight']  #del this col follow  census-income.name doc
del test_data['instance weight']

col_names.remove('instance weight')
label1, label2 = 'income', 'marital status'

col_names.remove(label1)
col_names.remove(label2)

#40 features, 7 dense, 33 sparse, label:income
continuous_cols = ['age', 'wage per hour', 'capital gains', 'capital losses', 'divdends from stocks', 'num persons worked for employer', 'weeks worked in year']
category_cols = [col for col in col_names if col not in continuous_cols]

n_train = train_data.shape[0]
data = pd.concat([train_data, test_data], axis=0)

for col in category_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])

sca = MinMaxScaler()
data[continuous_cols] = sca.fit_transform(data[continuous_cols])
data[continuous_cols] = data[continuous_cols].round(4)  #精度截断

#In MMOE PLE Paper： income prediction is the main task，married prediction is auxiliary task.
data[label1] = data[label1].map({" 50000+.": 1, ' - 50000.': 0})
data[label2] = data[label2].apply(lambda x: 1 if x == ' Never married' else 0)

#In origin paper, split val in test data by 1:1
df_train = data.iloc[:n_train, :]
df_test = data.iloc[n_train:, :]
target = df_test["income"]

X_val, X_test = train_test_split(df_test, test_size=0.5, stratify=target)

df_train.to_csv("./census_income_train.csv", index=False)
X_val.to_csv("./census_income_val.csv", index=False)
X_test.to_csv("./census_income_test.csv", index=False)