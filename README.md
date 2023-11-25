# song-recommendation-engine (Carlors good idea_KNN)
import pandas as pd
from sklearn.neighbors import NearestNeighbors

# 1. 加载数据（Download Data）
data = pd.read_csv("song_dataset.csv")  # （instead of）

# 2. 创建用户-歌曲矩阵（Create user-song matrix）
user_song_matrix = data.pivot(index='user', columns='song', values='listen_count').fillna(0)

# 3. 初始化KNN模型（Initialize the KNN model）
knn_model = NearestNeighbors(n_neighbors=5, metric='cosine', algorithm='brute')  # 使用余弦相似度

# 4. 训练模型（train）
knn_model.fit(user_song_matrix)

# 5. 定义函数进行推荐
def recommend(user_id, user_songs):
    # 获取用户听过的歌曲
    user_songs_matrix = user_song_matrix.loc[user_id].values.reshape(1, -1)
    user_songs_matrix[:, user_song_matrix.columns.isin(user_songs)] = 0  # 将用户听过的歌曲在矩阵中设为0

    # 寻找最近邻居（find Neighbor）
    distances, indices = knn_model.kneighbors(user_songs_matrix)

    # 获取最近邻居的歌曲 
    similar_user_songs = user_song_matrix.iloc[indices[0]].values

    # 找到用户还未听过的歌曲
    recommendations = set(user_song_matrix.columns) - set(user_songs)

    return list(recommendations)

# 6. 使用推荐函数
user_id = 'some_user_id'  # 
user_songs_listened = ['song1', 'song2', 'song3']  # 请替换为用户已经听过的歌曲列表
recommendations = recommend(user_id, user_songs_listened)

# 7. 输出推荐
print(f"Recommended Songs for User {user_id} (excluding listened songs):")
for song in recommendations:
    print(song)
