from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler


FEATURES = ["danceability", "energy", "speechiness", "instrumentalness", "acousticness", "valence"]


def load_songs():
    columns = ["track_id", "track_name", "track_artist", "track_popularity",
               "playlist_name", "playlist_genre", *FEATURES]
    raw = pd.read_csv(Path(__file__).with_name("spotify_songs.csv"), usecols=columns)
    party = raw.playlist_name.str.contains("party", case=False, na=False)
    songs = raw.drop_duplicates("track_id").set_index("track_id")
    songs["in_party_playlist"] = party.groupby(raw.track_id).any()
    songs["party_score"] = songs.danceability + songs.energy - songs.acousticness
    return songs


def build_mix(songs, guests=10, songs_per_guest=25, top_n=10, seed=42, standardized=False):
    pool = songs.reset_index().sort_values(
        ["track_popularity", "track_id"], ascending=[False, True], kind="mergesort"
    ).head(300).set_index("track_id")
    rng = np.random.default_rng(seed)
    picks = {}
    for i in range(guests):
        sample = pool.sample(songs_per_guest, random_state=int(rng.integers(1_000_000)))
        picks[f"Guest {i + 1}"] = sample.nlargest(top_n, "party_score")

    candidates = pd.concat(picks.values())
    candidates = candidates.loc[~candidates.index.duplicated()].copy()
    candidates = score_candidates(candidates, standardized)
    return picks, candidates, candidates.nlargest(25, "mean_sim")


def score_candidates(candidates, standardized):
    candidates = candidates.copy()
    features = candidates[FEATURES]
    if standardized:
        features = StandardScaler().fit_transform(features)
    similarity = cosine_similarity(features)
    np.fill_diagonal(similarity, np.nan)
    candidates["mean_sim"] = np.nanmean(similarity, axis=1) if len(candidates) > 1 else np.nan
    return candidates
