"""Deployed model"""


import os


from flask_wtf import FlaskForm as Form
from flask import Flask, render_template, request
import pandas as pd
from wtforms import SelectField, SubmitField


from utils import get_artists, get_song_list, get_recommendations

app = Flask(__name__)
SECRET_KEY = os.urandom(32)
app.config['SECRET_KEY'] = SECRET_KEY


song_df = pd.read_csv("song_dataset.csv")
songs = []
favourite_songs = set()
recommendations = {}


class Songs(Form):
    """Song data form"""
    artists = SelectField("select1", choices=get_artists(song_df))
    songs = SelectField("select2")
    submit = SubmitField("add")
    clear = SubmitField("clear")
    analyse = SubmitField("get recommendations")


@app.route("/", methods=["GET", "POST"])
def recommendation():
    """main page"""
    form = Songs(request.form)
    selected_artist = ""

    if request.method == "POST" and form.artists.data != selected_artist:
        selected_artist = form.artists.data
        form.songs.choices = get_song_list(song_df, form.artists.data)

    if form.is_submitted() and form.clear.data:
        favourite_songs.clear()
        recommendations.clear()

    if form.is_submitted() and form.submit.data:
        favourite_songs.add(
            (
                form.songs.data,
                selected_artist,
                dict(form.songs.choices).get(form.songs.data),
            ),
        )

    if form.is_submitted() and form.analyse.data:
        recommendations.update(get_recommendations(favourite_songs))

    return render_template(
        "home.html",
        form=form,
        song_list=list(favourite_songs),
        recommendations=recommendations,
    )


if __name__ == "__main__":
    app.run(debug=True)
