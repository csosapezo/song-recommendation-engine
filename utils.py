"""Util functions for the recommender"""

import pandas as pd


def get_songs_from_user(user_id: str, df: pd.DataFrame) -> list[str]:
    """get all songs a user has listened"""
    return df.loc[df.user == user_id].sort_values(
        "play_count",
        ascending=False,
    ).song.to_list()
