"""main function"""


import sys


from recommender import Recommender
from utils import get_songs_from_user


def get_recommendation_from_user(
        song_id: str,
        ) -> tuple[str, str, str, float, str]:
    """main function"""

    rcm = Recommender()
    rcm.get_sum_song_distribution()

    song_list: list[str] = get_songs_from_user(song_id, rcm.df)
    recommended: list[tuple[str, str, str, float, str]] = []

    for song in song_list:
        recommended.extend(rcm.recommend(song, song_list))

    recommended = sorted(recommended, key=lambda x: x[3], reverse=True)[0]
    ref_artist, ref_song = rcm.find_song_data(recommended[4])

    print(
        f"Listen to {recommended[1]} by {recommended[2]} because you"
        f" listened to {ref_artist} by {ref_song}"
    )


if __name__ == "__main__":
    get_recommendation_from_user(sys.argv[1])
