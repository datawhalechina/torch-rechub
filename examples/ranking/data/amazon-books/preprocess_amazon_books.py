import pandas as pd
from sklearn.preprocessing import LabelEncoder


def main(data_path):
    data = pd.read_csv(data_path, names=['user_id', 'item_id', 'rating', 'time'])
    data['item_count'] = data.groupby('item_id')['item_id'].transform('count')
    data = data[data.item_count >= 5]

    data['user_mean'] = data.groupby(by='user_id')['rating'].transform('mean')
    data['item_mean'] = data.groupby(by='item_id')['rating'].transform('mean')

    data.loc[(data.rating>=data.user_mean), 'label'] = 1
    data.loc[(data.rating<data.user_mean), 'label'] = 0

    data = data[['user_id', 'item_id', 'time', 'label']]
    for feat in data:
        le = LabelEncoder()
        data[feat] = le.fit_transform(data[feat])

    data = data.sort_values(by=['user_id'])
    data.to_csv('amazon_books_datasets.csv', index=False)
    data.head(100).to_csv('amazon_books_sample.csv', index=False)
    print(data)


if __name__ == '__main__':
    main('./ratings_Books.csv')
