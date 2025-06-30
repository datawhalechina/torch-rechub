import sqlite3

import pandas as pd
from sklearn.preprocessing import LabelEncoder
'''
这里的行数设定为500000行作为sample数据，实际数据非常大，全量训练（32G内存）可以尝试用3000000行
'''
data = pd.read_csv(r'train_triplets.txt', sep='\t', header=None, names=['user', 'song', 'play_count'], nrows=500000)

# label编码
user_encoder = LabelEncoder()
data['user'] = user_encoder.fit_transform(data['user'].values)

song_encoder = LabelEncoder()
data['song'] = song_encoder.fit_transform(data['song'].values)

# 数据类型转换
data.astype({'user': 'int32', 'song': 'int32', 'play_count': 'int32'})

# 用户的歌曲播放总量的分布
# 字典user_play_counts记录每个用户的播放总量
user_play_counts = {}
for user, group in data.groupby('user'):
    user_play_counts[user] = group['play_count'].sum()

temp_user = [user for user in user_play_counts.keys() if user_play_counts[user] > 100]
# 过滤掉歌曲播放量少于100的用户的数据
data = data[data.user.isin(temp_user)]

# song_play_counts字典，记录每首歌的播放量
song_play_counts = {}
for song, group in data.groupby('song'):
    song_play_counts[song] = group['play_count'].sum()

temp_song = [song for song in song_play_counts.keys() if song_play_counts[song] > 50]
# 过滤掉播放量小于50的歌曲
data = data[data.song.isin(temp_song)]

# 读取数据
conn = sqlite3.connect(r'.\track_metadata.db')
cur = conn.cursor()
cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
cur.fetchall()

# 获得数据的dataframe
track_metadata_df = pd.read_sql(con=conn, sql='select * from songs')
# 对于之前的歌曲编码，我们给一个字典，对歌曲和编码进行一一映射
song_labels = dict(zip(song_encoder.classes_, range(len(song_encoder.classes_))))

# 对于那些在之前没有出现过的歌曲，我们直接给一个最大的编码


def encoder(x):
    return song_labels[x] if x in song_labels.keys() else len(song_labels)


# 对数据进行labelencoder
track_metadata_df['song_id'] = track_metadata_df['song_id'].apply(encoder)
# 对song_id重命名为song
track_metadata_df = track_metadata_df.rename(columns={'song_id': 'song'})
# 根据特征song进行拼接，将拼接后的数据重新命名为data
data = pd.merge(data, track_metadata_df, on='song')
data = data.astype({'play_count': 'int32', 'duration': 'float32', 'artist_familiarity': 'float32', 'artist_hotttnesss': 'float32', 'year': 'int32', 'track_7digitalid': 'int32'})
# 去重
data.drop_duplicates(inplace=True)
'''
为了进一步精简数据，这里删掉了部分列，可根据个人需要删除列
'''
data.drop(['track_id', 'artist_id', 'artist_mbid', 'duration', 'track_7digitalid', 'shs_perf', 'shs_work'], axis=1, inplace=True)

data.to_csv("./million-song-dataset_sample.csv", index=False)
data.info()
