"""Util functions for the recommender"""

import pandas as pd


from recommender import Recommender


def get_songs_from_user(user_id: str, df: pd.DataFrame) -> list[str]:
    """get all songs a user has listened"""
    return df.loc[df.user == user_id].sort_values(
        "play_count",
        ascending=False,
    ).song.to_list()


def get_artists(df: pd.DataFrame) -> list[str]:
    """get all artist in the database"""
    artists = df[["artist_name"]].drop_duplicates().sort_values("artist_name")

    return artists["artist_name"].tolist()


def get_song_list(
        df: pd.DataFrame,
        artist_name: str,
        ) -> list[tuple[str, str, str]]:
    """get all songs from an artist in the dataset"""

    song_list = df.loc[df.artist_name == artist_name][[
        "song", "title"]].drop_duplicates().sort_values("title")

    return list(song_list.itertuples(index=False))


def get_recommendations(song_list: set[tuple[str, str, str]]) -> dict:
    """get recommendations from user favourite songs"""
    recommender = Recommender()
    recommender.get_sum_song_distribution()
    recommendations = {}

    songs_id = [song[0] for song in song_list]

    for song in song_list:
        recommended_songs = recommender.recommend(song[0], songs_id, 5)
        recommendations[song] = recommended_songs

    return recommendations
