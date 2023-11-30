"""recommender functions"""

import attrs
import pandas as pd


@attrs.define
class Recommender:
    """recommender"""
    df: pd.DataFrame = pd.read_csv("song_dataset.csv")
    dist_df: pd.DataFrame = None

    def get_song_distribution(self) -> pd.DataFrame:
        """get distribution DataFrame"""
        self.dist_df = self.df.groupby([
            "song",
            "user",
        ]).count()["play_count"].unstack(level=0).fillna(0)

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
        recommended_df = recommended_df.loc[recommended_df > 0]

        return recommended_df.sort_values(ascending=False).head()
