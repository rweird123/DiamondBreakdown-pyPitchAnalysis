import streamlit as st
from pybaseball import statcast_pitcher, playerid_lookup
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

st.set_page_config(page_title="Diamond Breakdown | Pitcher Velocity", layout="wide")
st.title("⚾ Diamond Breakdown — Pitcher Velocity Dashboard")
st.caption("Velocity trends, season baseline, and injury risk flagging powered by MLB Statcast.")

# USER INPUTS - pitcher name, time, pitches

col1, col2, col3 = st.columns(3)
with col1:
    player_name = st.text_input("Pitcher Name", "Gerrit Cole",
                                help="Use 'First Last' format (e.g. Justin Steele)")
with col2:
    season_year = st.selectbox(
        "Season",
        [2024, 2023, 2022, 2021, 2020],
        index=0
    )
with col3:
    st.markdown(f"**Season:** {season_year}")

# Convert season to date range (same behavior as before, just automated)
start_date = pd.to_datetime(f"{season_year}-04-01")
end_date = pd.to_datetime(f"{season_year}-09-30")

col4, col5 = st.columns(2)

# can change the rolling average time frame

with col4:
    rolling_window = st.slider("Rolling Average Window (games)", 3, 10, 5)

# creates a risk threshold of how much velo dropped by over time frame

with col5:
    risk_threshold = st.slider("Risk Flag Threshold (mph drop)", 0.5, 3.0, 1.5, step=0.1)

# PITCH NAME MAPPING - convert baseball savant pitch names to real pitches

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
    "SC": "Screwball",
}
pitch_code_map = {v: k for k, v in pitch_name_map.items()}

selected_pitches = st.multiselect(
    "Pitch Types to Display",
    list(pitch_name_map.values()),
    default=["Four-Seam Fastball", "Slider"],
)

# DATA FETCHING (CACHED) - fetch from statcast, save data so it works faster

@st.cache_data(show_spinner="Fetching Statcast data...")
def get_pitcher_data(player_name: str, start_date: str, end_date: str):
    parts = player_name.strip().split()
    if len(parts) < 2:
        return None, None

    # Handle suffixes like Jr., Sr., III
    first, last = parts[0], parts[-1]
    lookup = playerid_lookup(last, first)
    if lookup.empty:
        return None, None

    player_id = lookup["key_mlbam"].iloc[0]
    data = statcast_pitcher(start_date, end_date, player_id)
    return data, player_id

# MAIN DASHBOARD - error message added

if st.button("Generate Dashboard", type="primary"):

    with st.spinner("Loading..."):
        pitcher_df, player_id = get_pitcher_data(
            player_name, str(start_date), str(end_date)
        )

    if pitcher_df is None:
        st.error("Couldn't find that player. Check spelling and use 'First Last' format.")
        st.stop()

    if pitcher_df.empty:
        st.error("No Statcast data found for that player/date range.")
        st.stop()

    pitcher_df["game_date"] = pd.to_datetime(pitcher_df["game_date"])
    full_dates = pd.date_range(start=start_date, end=end_date)
    full_df = pd.DataFrame({"game_date": full_dates})

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    risk_report = []
    colors = ["#4FC3F7", "#FF8A65", "#A5D6A7", "#CE93D8", "#FFF176"]

    for i, pitch_full_name in enumerate(selected_pitches):
        pitch_code = pitch_code_map[pitch_full_name]
        pitch_df = pitcher_df[pitcher_df["pitch_type"] == pitch_code].copy()

        if pitch_df.empty:
            st.warning(f"No data found for {pitch_full_name} in this date range.")
            continue

        pitch_df = pitch_df.rename(columns={"release_speed": "velo"})
        pitch_df["velo"] = pd.to_numeric(pitch_df["velo"], errors="coerce")
        pitch_df = pitch_df.dropna(subset=["velo"])

        # Per-game average
        game_avg = pitch_df.groupby("game_date")["velo"].mean().reset_index()
        game_avg = pd.merge(full_df, game_avg, on="game_date", how="left")
        game_avg["velo_interp"] = game_avg["velo"].interpolate()
        game_avg["velo_rolling"] = (
            game_avg["velo_interp"]
            .rolling(rolling_window, min_periods=1)
            .mean()
        )

        color = colors[i % len(colors)]

        # Season baseline (median of per-game averages)
        season_baseline = game_avg["velo"].median()

        # Raw velocity (dashed, transparent)
        ax.plot(
            game_avg["game_date"],
            game_avg["velo"],
            alpha=0.2,
            linestyle="--",
            color=color,
        )

        # Rolling average line
        ax.plot(
            game_avg["game_date"],
            game_avg["velo_rolling"],
            label=f"{pitch_full_name} (rolling avg)",
            color=color,
            linewidth=2,
        )

        # Baseline reference line
        ax.axhline(
            season_baseline,
            linestyle=":",
            linewidth=1.2,
            color=color,
            alpha=0.6,
            label=f"{pitch_full_name} baseline ({season_baseline:.1f} mph)",
        )

        # risk flagging - if it drops bellow baseline

        # Only look at rows where we have actual game data (not interpolated gaps)
        flagged = game_avg.dropna(subset=["velo"]).copy()
        flagged["drop_from_baseline"] = season_baseline - flagged["velo_rolling"]
        risk_games = flagged[flagged["drop_from_baseline"] >= risk_threshold]

        if not risk_games.empty:
            # Shade risk zones
            for _, row in risk_games.iterrows():
                ax.axvspan(
                    row["game_date"] - pd.Timedelta(days=2),
                    row["game_date"] + pd.Timedelta(days=2),
                    color="red",
                    alpha=0.12,
                )
            # Mark risk points
            ax.scatter(
                risk_games["game_date"],
                risk_games["velo_rolling"],
                color="red",
                zorder=5,
                s=60,
                label=f"⚠ Risk zone ({pitch_full_name})" if i == 0 else "",
            )

            # Build risk report text
            risk_report.append(f"**{pitch_full_name}**")
            for _, row in risk_games.iterrows():
                drop = row["drop_from_baseline"]
                risk_report.append(
                    f"  • {row['game_date'].strftime('%b %d')}: "
                    f"{row['velo_rolling']:.1f} mph — "
                    f"**{drop:.1f} mph below baseline** ⚠️"
                )

    # chart styling to look tuff

    ax.set_title(f"{player_name} — Velocity Trends", color="white", fontsize=14, pad=12)
    ax.set_xlabel("Date", color="#aaaaaa")
    ax.set_ylabel("Velocity (mph)", color="#aaaaaa")
    ax.tick_params(colors="#aaaaaa")
    ax.spines["bottom"].set_color("#333333")
    ax.spines["left"].set_color("#333333")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, color="#222222", linewidth=0.5)

    legend = ax.legend(
        facecolor="#1a1a2e", edgecolor="#333333", labelcolor="white", fontsize=8
    )

    red_patch = mpatches.Patch(color="red", alpha=0.4, label=f"Risk zone (≥{risk_threshold} mph drop)")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles=handles + [red_patch],
        facecolor="#1a1a2e",
        edgecolor="#333333",
        labelcolor="white",
        fontsize=8,
    )

    st.pyplot(fig)

    # risk report generating
    st.divider()
    st.subheader("🚨 Risk Report")

    if risk_report:
        st.markdown(
            f"Games where rolling average dropped **≥ {risk_threshold} mph** below season baseline:"
        )
        for line in risk_report:
            st.markdown(line)
    else:
        st.success(
            f"✅ No risk zones detected. Rolling velocity stayed within "
            f"{risk_threshold} mph of the season baseline throughout this period."
        )
