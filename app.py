import pandas as pd
import streamlit as st

from vibe import build_mix, load_songs, score_candidates


st.set_page_config(page_title="Hitting the Vibe")
st.title("Hitting the Vibe")
st.write("Build a party playlist from simulated guests' favorites among the 300 most popular tracks.")

with st.sidebar:
    st.header("Party settings")
    guests = st.slider("Number of guests", 3, 15, 10)
    per_guest = st.number_input("Songs per guest", 1, 300, 25)
    top_n = st.number_input("Top tracks per guest", 1, per_guest, min(10, per_guest))
    seed = st.number_input("Random seed", 0, 2**32 - 1, 42)
    standardized = st.toggle("Standardize features", value=False)

songs = st.cache_data(load_songs)()
picks, candidates, mix = build_mix(songs, guests, per_guest, top_n, seed, standardized)
labels = {"track_name": "Title", "track_artist": "Artist", "playlist_genre": "Genre",
          "party_score": "Party score", "mean_sim": "Mean similarity"}


def show_tracks(frame, columns):
    table = frame[columns].rename(columns=labels)
    st.dataframe(table, hide_index=True, column_config={
        name: st.column_config.NumberColumn(format="%.3f")
        for name in ("Party score", "Mean similarity") if name in table
    })


st.header("Party playlist")
st.caption(f"{len(mix)} tracks from {len(candidates)} unique guest picks, ranked by mean similarity.")
if len(candidates) < 25:
    st.info("Fewer than 25 distinct tracks qualified, so every candidate is included.")
if len(candidates) == 1:
    st.caption("Mean similarity needs at least two distinct tracks.")
show_tracks(mix, list(labels))

st.subheader("Genre breakdown")
counts = mix.playlist_genre.value_counts()
st.bar_chart(counts.rename_axis("Genre").rename("Tracks"))

st.header("Guest picks")
st.caption("Each guest contributes their highest-scoring tracks. Shared picks enter the candidate pool once.")
for guest, tracks in picks.items():
    with st.expander(f"{guest} · {len(tracks)} tracks"):
        show_tracks(tracks, ["track_name", "track_artist", "party_score"])

st.header("What we learned")
st.caption("A snapshot of this party. Change the settings to explore another mix.")

votes = pd.concat(picks.values()).index.value_counts()
leaders = votes[votes == votes.max()].index
st.subheader("Most shared guest pick")
if votes.max() == 1:
    st.write("No two guests shortlisted the same song.")
else:
    top = candidates.loc[leaders[0]]
    st.text(f"{top.track_name} — {top.track_artist}")
    st.write(f"Shortlisted by {votes.max()} of {guests} guests.")
    if len(leaders) > 1:
        with st.expander(f"All {len(leaders)} tracks tied for first"):
            show_tracks(candidates.loc[leaders], ["track_name", "track_artist"])

st.subheader("Sound of the final playlist")
genres = ", ".join(counts[counts == counts.max()].index)
if (counts == counts.max()).sum() == 1:
    st.write(f"{genres} leads with {counts.max()} of {len(mix)} tracks.")
else:
    st.write(f"Tied for the lead at {counts.max()} tracks each: {genres}.")

other_mix = score_candidates(candidates, not standardized).nlargest(25, "mean_sim")
changed = len(mix.index.difference(other_mix.index))
st.subheader("Same guests, different scoring")
st.write(f"Switching feature standardization {'off' if standardized else 'on'} "
         f"would replace {changed} of {len(mix)} tracks.")
