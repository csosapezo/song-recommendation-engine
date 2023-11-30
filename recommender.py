"""recommender functions"""

import attrs
import numpy as np
import pandas as pd
from scipy import stats


@attrs.define
class Recommender:
    """recommender"""
    df: pd.DataFrame = pd.read_csv("song_dataset.csv")
    sum_dist_df: pd.DataFrame = None

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
        """get a list of new songs the user has never listened to
        according to a song"""

        similar_users_df: pd.DataFrame
        similar_users_df = self.sum_dist_df.loc[self.sum_dist_df[song_id] > 0]
        similar_users_df = similar_users_df.multiply(
            similar_users_df[song_id], axis="index"
        )
        similar_users_df = similar_users_df.drop(song_list, axis=1)

        recommended: pd.Series = similar_users_df.sum()

        return stats.zscore(recommended)

    def recommend_from_artist(
            self,
            song_id: str,
            song_list: list[str]
            ) -> tuple[pd.Series, pd.Series]:
        """get a list of new songs the user has never listened to
        according to the artist of a song"""

        artist_songs: np.ndarray = self.get_songs_from_artist(song_id)

        similar_users_df: pd.DataFrame
        similar_users_df = self.sum_dist_df.loc[
            (self.sum_dist_df[artist_songs] > 0).any(axis=1)
        ]

        songs_not_from_artist = list(
            set(similar_users_df.columns) - set(artist_songs)
        )
        similar_users_df[songs_not_from_artist] = 0

        similar_users_df["weight"] = similar_users_df.sum(axis=1)
        similar_users_df = similar_users_df.multiply(
            similar_users_df["weight"] / len(artist_songs), axis="index"
        )
        similar_users_df = similar_users_df.drop("weight", axis=1)

        similar_users_df = similar_users_df.drop(song_list, axis=1)

        recommended: pd.Series = similar_users_df.sum()

        is_from_artist: pd.Series = pd.Series(
            [1 if x in artist_songs else 0 for x in recommended.index],
            index=recommended.index,
        )

        return stats.zscore(recommended), is_from_artist

    def find_song_data(self, song_id: str) -> tuple[str, str]:
        """get a song's title and artist from an ID"""

        song = self.df.loc[self.df.song == song_id].sample()

        return song.title, song.artist_name

    def recommend(
            self,
            song_id: str,
            song_list: list[str],
            num_recommendation: int = 1,
            ) -> list[tuple[str, str, str, float]]:
        """recommend new songs based on a song the user has listened"""

        recommended_by_song: pd.Series = self.recommend_from_song(
            song_id,
            song_list,
        )

        recommended_by_artist: pd.Series
        is_from_artist: pd.Series
        recommended_by_artist, is_from_artist = self.recommend_from_artist(
            song_id,
            song_list,
        )

        rec: pd.Series = ((
            0.6 * recommended_by_song
            + 0.4 * recommended_by_artist * is_from_artist
            ) / (0.6 + 0.4 * is_from_artist)
        ).sort_values(ascending=False).head(num_recommendation)

        top_recommendations: list = [
            (idx, *self.find_song_data(idx), rec.loc[idx])
            for idx in rec.index
        ]

        return top_recommendations
