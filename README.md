# Hitting the Vibe: a group party playlist from Spotify audio features

A small project, and my first hands-on work with data. It started as a four-person course project in *Data Analysis in the Music Industry* at NYU (Spring 2024; team: Fred Xiang, Owen Tynes, Sam Richter, Gerrit Wendt). In November 2026 I re-implemented the algorithm on my own, tested the team's conclusions with data, and built this small interactive demo.

**The question:** how do you turn ten guests' favorite songs into one party playlist, and can Spotify's audio features tell which songs are "party" songs?

## How the playlist is built

1. Each simulated guest gets 25 songs drawn from the 300 most popular tracks.
2. Each song gets a party score: danceability + energy − acousticness. Each guest contributes their top 10.
3. After removing duplicates, every candidate is compared with all the others by cosine similarity over six audio features (danceability, energy, speechiness, instrumentalness, acousticness, valence).
4. The 25 songs with the highest average similarity form the playlist.

## What the re-implementation found

From [`hitting_the_vibe_reimplementation.ipynb`](hitting_the_vibe_reimplementation.ipynb), on about 28,000 songs:

- **A threshold cannot pick out party songs, which confirms the team's choice to rank instead.** Using "appears in a playlist with *party* in its name" as a rough label, danceability scores an AUC of 0.48 (no better than chance) and the party score 0.60.
- **Audio features carry some genre information, but not enough to separate genres.** A random forest on the six features reaches 47% accuracy on six genres, against a 19% majority-class baseline.
- **The playlist depends heavily on preprocessing.** On raw features every candidate's average similarity is above 0.88; standardizing the features first replaces 10 of the 25 songs.
- **Ties need a rule.** 57 songs share the popularity value at the top-300 cutoff, so without a tie-break the pool, and every result after it, differs between machines. The pool is sorted by popularity, then by track ID.

## Run it

```sh
python -m pip install -r requirements.txt
curl -L -o spotify_songs.csv https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-01-21/spotify_songs.csv
python -m streamlit run app.py
```

The sidebar sets the number of guests, songs per guest, top tracks per guest, the random seed, and whether features are standardized. With the defaults (10 guests, seed 42) you should see 77 candidates and a playlist of 15 pop, 5 rap and 5 latin tracks.

## Files

| File | What it does |
|---|---|
| `vibe.py` | Loads the data and builds the playlist |
| `app.py` | Streamlit interface |
| `hitting_the_vibe_reimplementation.ipynb` | Step-by-step re-implementation and the tests above (notes in Chinese) |

## Data and limits

- Data: [TidyTuesday 2020-01-21 Spotify songs](https://github.com/rfordatascience/tidytuesday/tree/master/data/2020/2020-01-21), collected with the `spotifyr` package. Not redistributed here. Spotify's audio-features API is closed to new apps, so the original live-API design is replaced by this static dataset.
- Guests are simulated, not real users; nothing here measures whether people enjoy the playlist.
- The "party" label comes from playlist names and the genre label from the playlist's genre, so both are rough.
