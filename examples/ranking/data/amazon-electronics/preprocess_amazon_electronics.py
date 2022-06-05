import pandas as pd

def json_to_df(file_path):
    with open(file_path, 'r') as f:
        df = {}
        i = 0
        for line in f:
            df[i] = eval(line)
            i += 1
        df = pd.DataFrame.from_dict(df, orient='index')
        return df


def build_map(df, col_name):
    key = sorted(df[col_name].unique().tolist())
    m = dict(zip(key, range(len(key))))
    df[col_name] = df[col_name].map(lambda x: m[x])
    return m, key


def parse_data_to_df(reviews_file_path, meta_file_path):
    print('========== Start reading data ==========')
    reviews_df = json_to_df(reviews_file_path)
    reviews_df = reviews_df[['reviewerID', 'asin', 'unixReviewTime']]

    meta_df = json_to_df(meta_file_path)
    meta_df = meta_df[meta_df['asin'].isin(reviews_df['asin'].unique())]
    meta_df = meta_df.reset_index(drop=True)
    meta_df = meta_df[['asin', 'categories']]
    meta_df['categories'] = meta_df['categories'].map(lambda x: x[-1][-1])  # Category features keep only one
    print('========== DataFrame file successfully read ==========')

    asin_map, asin_key = build_map(meta_df, 'asin')
    build_map(meta_df, 'categories')
    build_map(reviews_df, 'reviewerID')

    meta_df = meta_df.sort_values('asin')
    meta_df = meta_df.reset_index(drop=True)

    reviews_df['asin'] = reviews_df['asin'].map(lambda x: asin_map[x])
    reviews_df = reviews_df.sort_values(['reviewerID', 'unixReviewTime'])
    reviews_df = reviews_df.reset_index(drop=True)
    reviews_df = reviews_df[['reviewerID', 'asin', 'unixReviewTime']]

    data = pd.merge(reviews_df, meta_df, how='inner', on='asin')
    data.rename(columns={'asin': 'item_id',
                         'reviewerID': 'user_id',
                         'unixReviewTime': 'time',
                         'categories': 'cate_id'}, inplace=True)

    return data
