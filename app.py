import streamlit as st
from pybaseball import statcast_pitcher, playerid_lookup
import pandas as pd
import matplotlib.pyplot as plt

st.title("⚾ Pitcher Velocity Dashboard")

# inputs (name, pitches)
player_name = st.text_input("Enter Pitcher Name", "Gerrit Cole")

col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Start Date", pd.to_datetime("2024-04-01"))
with col2:
    end_date = st.date_input("End Date", pd.to_datetime("2024-09-30"))

pitch_types = st.multiselect(
    "Select Pitch Types",
    ["FF", "SI", "SL", "CH", "CU", "FC"],
    default=["FF", "SL"]
)

# convert pitch abbreviations to real pitches
pitch_name_map = {
    "FF": "Four-Seam Fastball",
    "SI": "Sinker",
    "SL": "Slider",
    "CH": "Changeup",
    "CU": "Curveball",
    "FC": "Cutter",
    "KC": "Knuckle Curve",
    "FS": "Splitter",
    "FO": "Forkball",
    "EP": "Eephus",
    "KN": "Knuckleball",
    "SC": "Screwball"
}

# get pitcher data
def get_pitcher_data(player_name, start_date, end_date):
    df = playerid_lookup(player_name.split()[1], player_name.split()[0])
    if df.empty:
        return None, None
    player_id = df['key_mlbam'].iloc[0]
    data = statcast_pitcher(str(start_date), str(end_date), player_id)
    return data, player_id

# generate dashboard
if st.button("Generate Dashboard"):

    pitcher_df, player_id = get_pitcher_data(player_name, start_date, end_date)

    if pitcher_df is None or pitcher_df.empty:
        st.error("No data found for that player/date range.")
    else:
        pitcher_df["game_date"] = pd.to_datetime(pitcher_df["game_date"])

        # fixed date range for x-axis
        full_dates = pd.date_range(start=start_date, end=end_date)
        full_df = pd.DataFrame({"game_date": full_dates})

        fig, ax = plt.subplots(figsize=(12,6))

        for pitch_code in pitch_types:
            pitch_df = pitcher_df[pitcher_df["pitch_type"] == pitch_code].copy()

            if pitch_df.empty:
                st.warning(f"No data for {pitch_name_map.get(pitch_code, pitch_code)}")
                continue

            pitch_df = pitch_df.rename(columns={"release_speed": "velo"})
            pitch_df["velo"] = pd.to_numeric(pitch_df["velo"], errors="coerce")
            pitch_df = pitch_df.dropna(subset=["velo"])

            # average per game
            game_avg = pitch_df.groupby("game_date")["velo"].mean().reset_index()

            # merge with full date range for consistent x-axis
            game_avg = pd.merge(full_df, game_avg, on="game_date", how="left")

            # skip missing values
            game_avg["velo_interp"] = game_avg["velo"].interpolate()

            # create rolling average
            game_avg["velo_rolling"] = game_avg["velo_interp"].rolling(5, min_periods=1).mean()

            # plot raw velocity (dashed, transparent)
            ax.plot(
                game_avg["game_date"],
                game_avg["velo"],
                alpha=0.25,
                linestyle="--"
            )

            # plot rolling/smoothed velocity
            ax.plot(
                game_avg["game_date"],
                game_avg["velo_rolling"],
                label=pitch_name_map.get(pitch_code, pitch_code)
            )

        ax.set_title(f"{player_name} Velocity Trends")
        ax.set_xlabel("Date")
        ax.set_ylabel("Velocity (mph)")
        ax.legend(title="Pitch Type")
        ax.grid(True)

        st.pyplot(fig)
