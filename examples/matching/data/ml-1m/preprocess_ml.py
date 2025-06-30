import pandas as pd

data_path = "./data/"

unames = ['user_id', 'gender', 'age', 'occupation', 'zip']
user = pd.read_csv(data_path + 'ml-1m/users.dat', sep='::', header=None, names=unames, engine='python', encoding="ISO-8859-1")
rnames = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_csv(data_path + 'ml-1m/ratings.dat', sep='::', header=None, names=rnames, engine='python', encoding="ISO-8859-1")
mnames = ['movie_id', 'title', 'genres']
movies = pd.read_csv(data_path + 'ml-1m/movies.dat', sep='::', header=None, names=mnames, engine='python', encoding="ISO-8859-1")

data = pd.merge(pd.merge(ratings, movies), user)
data.to_csv("./data/ml-1m/ml-1m.csv", index=False)
