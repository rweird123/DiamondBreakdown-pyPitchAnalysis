import streamlit as st
from pybaseball import statcast_pitcher, playerid_lookup
import pandas as pd
import matplotlib.pyplot as plt

st.title("⚾ Pitcher Velocity Dashboard")

# --- USER INPUTS ---
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

# --- MAIN FUNCTION ---
def get_pitcher_data(player_name, start_date, end_date):
    df = playerid_lookup(player_name.split()[1], player_name.split()[0])

    if df.empty:
        return None, None

    player_id = df['key_mlbam'].iloc[0]
    data = statcast_pitcher(str(start_date), str(end_date), player_id)

    return data, player_id

# --- BUTTON ---
if st.button("Generate Dashboard"):

    pitcher_df, player_id = get_pitcher_data(player_name, start_date, end_date)

    if pitcher_df is None or pitcher_df.empty:
        st.error("No data found for that player/date range.")
    else:
        pitcher_df["game_date"] = pd.to_datetime(pitcher_df["game_date"])

        # FIXED DATE RANGE
        full_dates = pd.date_range(start=start_date, end=end_date)
        full_df = pd.DataFrame({"game_date": full_dates})

        fig, ax = plt.subplots(figsize=(12,6))

        for pitch_type in pitch_types:
            pitch_df = pitcher_df[pitcher_df["pitch_type"] == pitch_type].copy()

            if pitch_df.empty:
                st.warning(f"No data for {pitch_type}")
                continue

            pitch_df = pitch_df.rename(columns={"release_speed": "velo"})
            pitch_df["velo"] = pd.to_numeric(pitch_df["velo"], errors="coerce")
            pitch_df = pitch_df.dropna(subset=["velo"])

            game_avg = pitch_df.groupby("game_date")["velo"].mean().reset_index()

            # Merge with full date range
            game_avg = pd.merge(full_df, game_avg, on="game_date", how="left")

            # Interpolate
            game_avg["velo_interp"] = game_avg["velo"].interpolate()

            # Rolling avg
            game_avg["velo_rolling"] = game_avg["velo_interp"].rolling(5, min_periods=1).mean()

            # Plot raw
            ax.plot(
                game_avg["game_date"],
                game_avg["velo"],
                alpha=0.25,
                linestyle="--"
            )

            # Plot smooth
            ax.plot(
                game_avg["game_date"],
                game_avg["velo_rolling"],
                label=pitch_type
            )

        ax.set_title(f"{player_name} Velocity Trends")
        ax.set_xlabel("Date")
        ax.set_ylabel("Velocity (mph)")
        ax.legend(title="Pitch Type")
        ax.grid(True)

        st.pyplot(fig)

        # --- BONUS STATS ---
        st.subheader("📊 Quick Stats")

        for pitch_type in pitch_types:
            pitch_df = pitcher_df[pitcher_df["pitch_type"] == pitch_type]

            if not pitch_df.empty:
                avg_velo = pitch_df["release_speed"].mean()
                max_velo = pitch_df["release_speed"].max()
