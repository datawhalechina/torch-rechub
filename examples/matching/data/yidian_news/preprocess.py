import pandas as pd

# %%

ui_path = '~/Rec/data/YidianNews/train_data.txt'
user_path = '~/Rec/data/YidianNews/user_info.txt'

# fetch first 1000 user-item rows for example, no missing values or
# inconsistent format
ui_df = pd.read_csv(ui_path, header=None, sep='\t', nrows=1000, names=['userId', 'itemId', 'showTime', 'network', 'refresh', 'showPos', 'click', 'duration'])
user_df = pd.read_csv(user_path, header=None, sep='\t', names=['userId', 'deviceName', 'OS', 'province', 'city', 'age', 'gender'])

# %%
data = ui_df.merge(user_df, on='userId', how='left')

# %%
age = data.age
age_df = age.str.split(',', expand=True)
age_df.columns = ['age0', 'age1', 'age2', 'age3']
age_df = age_df.applymap(lambda x: float(x.split(':')[1]))

# %%
gender = data.gender
gender_df = gender.str.split(',', expand=True)
gender_df.columns = ['female', 'male']
gender_df = gender_df.applymap(lambda x: float(x.split(':')[1]))

# %%
data = pd.concat([data.drop(['age', 'gender', 'duration'], axis=1), age_df, gender_df], axis=1)
data.to_csv('yidian_news_sampled.csv')
print(data)
