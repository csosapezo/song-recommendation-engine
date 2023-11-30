"""recommender functions"""

import attrs
import numpy as np
import pandas as pd
from scipy import stats


@attrs.define
class Recommender:
    """recommender"""
    df: pd.DataFrame = pd.read_csv("song_dataset.csv")
    dist_df: pd.DataFrame = None
    sum_dist_df: pd.DataFrame = None

    def get_song_distribution(self) -> pd.DataFrame:
        """get distribution DataFrame"""
        self.dist_df = self.df.groupby([
            "song",
            "user",
        ]).count()["play_count"].unstack(level=0).fillna(0)

    def get_sum_song_distribution(self) -> pd.DataFrame:
        """get distribution DataFrame (considering play count)"""
        self.sum_dist_df = self.df.groupby([
            "song",
            "user",
        ]).sum()["play_count"].unstack(level=0).fillna(0)

    def get_songs_from_artist(self, song_id: str) -> np.ndarray:
        """get all the songs from an artist in the database"""

        artist_name: str
        artist_name = self.df.loc[self.df.song == song_id].artist_name.iloc[0]

        songs: pd.Series
        songs = self.df.loc[self.df.artist_name == artist_name].song

        return songs.unique()

    def recommend_from_song(
            self,
            song_id: str,
            song_list: list[str]
            ) -> pd.Series:
        """get a list of new songs the user has never listened"""

        similar_users_df: pd.DataFrame
        similar_users_df = self.dist_df.loc[self.dist_df[song_id] > 0]
        similar_users_df = similar_users_df.drop(song_list, axis=1)

        recommended_df: pd.Series = similar_users_df.sum()

        return stats.zscore(recommended_df)

    def recommend_from_artist(
            self,
            song_id: str,
            song_list: list[str]
            ) -> pd.Series:
        """get a list of new songs the user has never listened"""

        artist_songs: np.ndarray = self.get_songs_from_artist(song_id)

        similar_users_df: pd.DataFrame
        similar_users_df = self.sum_dist_df.loc[
            (self.sum_dist_df[artist_songs] > 0).any(axis=1)
        ]
        similar_users_df = similar_users_df.multiply(
            similar_users_df[song_id], axis="index"
        )
        similar_users_df = similar_users_df.drop(song_list, axis=1)

        recommended_df: pd.Series = similar_users_df.sum()

        return stats.zscore(recommended_df)
