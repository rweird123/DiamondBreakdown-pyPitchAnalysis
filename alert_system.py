"""
alert_system.py
DiamondBreakdown — Automated Pitcher Velocity Alert System

Runs daily Mon–Fri during MLB season. For each pitcher in the watchlist,
pulls the last 30 days of Statcast data, computes a rolling average, and
compares against the season baseline. If the rolling avg drops >= threshold,
fires a tweet and a text message.

Required environment variables (set as GitHub Actions secrets):
    TWITTER_API_KEY
    TWITTER_API_SECRET
    TWITTER_ACCESS_TOKEN
    TWITTER_ACCESS_SECRET
    TWILIO_ACCOUNT_SID
    TWILIO_AUTH_TOKEN
    TWILIO_FROM_NUMBER       (your Twilio phone number, e.g. +12345678900)
    TWILIO_TO_NUMBER         (your personal cell, e.g. +17738675309)
"""

import os
import json
from datetime import datetime, timedelta

import pandas as pd
import tweepy
from twilio.rest import Client
from pybaseball import statcast_pitcher, playerid_lookup

# ── CONFIG ────────────────────────────────────────────────────────────────────

WATCHLIST = [
    {"name": "Justin Steele",  "pitch_type": "FF"},
    {"name": "Gerrit Cole",    "pitch_type": "FF"},
    {"name": "Paul Skenes",  "pitch_type": "FF"},
    {"name": "George Kirby",    "pitch_type": "FF"},
    # Add more pitchers here — same format
]

ROLLING_WINDOW   = 5      # games
RISK_THRESHOLD   = 1.5    # mph below baseline to trigger alert
LOOKBACK_DAYS    = 30     # how far back to pull recent data
SEASON_START     = "2025-03-27"  # update each year
DASHBOARD_URL    = "https://diamondbreakdown-pypitchanalysis-gha4m6jdzzuuqovwifc8zg.streamlit.app/"
ALERT_LOG        = "alert_log.json"  # tracks already-sent alerts to avoid duplicates

# ── HELPERS ───────────────────────────────────────────────────────────────────

def load_alert_log() -> dict:
    if os.path.exists(ALERT_LOG):
        with open(ALERT_LOG) as f:
            return json.load(f)
    return {}


def save_alert_log(log: dict):
    with open(ALERT_LOG, "w") as f:
        json.dump(log, f, indent=2)


def already_alerted(log: dict, pitcher: str, week: str) -> bool:
    """Prevent duplicate alerts within the same week for the same pitcher."""
    return log.get(pitcher, {}).get("last_alert_week") == week


def mark_alerted(log: dict, pitcher: str, week: str):
    if pitcher not in log:
        log[pitcher] = {}
    log[pitcher]["last_alert_week"] = week


def get_pitcher_data(player_name: str, start_date: str, end_date: str):
    parts = player_name.strip().split()
    first, last = parts[0], parts[-1]
    lookup = playerid_lookup(last, first)
    if lookup.empty:
        print(f"  ⚠ Could not find player ID for {player_name}")
        return None
    player_id = lookup["key_mlbam"].iloc[0]
    data = statcast_pitcher(start_date, end_date, player_id)
    return data if not data.empty else None


def compute_rolling_avg(df: pd.DataFrame, pitch_type: str, window: int):
    pitch_df = df[df["pitch_type"] == pitch_type].copy()
    if pitch_df.empty:
        return None, None

    pitch_df["release_speed"] = pd.to_numeric(pitch_df["release_speed"], errors="coerce")
    pitch_df = pitch_df.dropna(subset=["release_speed"])
    pitch_df["game_date"] = pd.to_datetime(pitch_df["game_date"])

    game_avg = pitch_df.groupby("game_date")["release_speed"].mean().reset_index()
    game_avg = game_avg.sort_values("game_date")

    season_baseline = game_avg["release_speed"].median()
    game_avg["rolling"] = game_avg["release_speed"].rolling(window, min_periods=1).mean()

    latest_rolling = game_avg["rolling"].iloc[-1]
    return latest_rolling, season_baseline


# ── ALERTS ────────────────────────────────────────────────────────────────────

def send_tweet(message: str):
    client = tweepy.Client(
        consumer_key=os.environ["TWITTER_API_KEY"],
        consumer_secret=os.environ["TWITTER_API_SECRET"],
        access_token=os.environ["TWITTER_ACCESS_TOKEN"],
        access_token_secret=os.environ["TWITTER_ACCESS_SECRET"],
    )
    response = client.create_tweet(text=message)
    print(f"  ✅ Tweet sent (id: {response.data['id']})")


def send_text(message: str):
    client = Client(
        os.environ["TWILIO_ACCOUNT_SID"],
        os.environ["TWILIO_AUTH_TOKEN"],
    )
    msg = client.messages.create(
        body=message,
        from_=os.environ["TWILIO_FROM_NUMBER"],
        to=os.environ["TWILIO_TO_NUMBER"],
    )
    print(f"  ✅ Text sent (sid: {msg.sid})")


def build_tweet(pitcher_name: str, pitch_type: str, rolling: float,
                baseline: float, drop: float) -> str:
    pitch_labels = {
        "FF": "4-seam fastball", "SI": "sinker", "SL": "slider",
        "CH": "changeup", "CU": "curveball", "FC": "cutter",
    }
    pitch_label = pitch_labels.get(pitch_type, pitch_type)

    tweet = (
        f"⚠️ VELOCITY ALERT — {pitcher_name}'s {pitch_label} is sitting "
        f"{rolling:.1f} mph ({drop:.1f} mph below season baseline of {baseline:.1f} mph) "
        f"over the last {ROLLING_WINDOW} outings.\n\n"
        f"Injury risk flag raised 🚩\n\n"
        f"📊 {DASHBOARD_URL}\n\n"
        f"#MLB #Statcast #DiamondBreakdown"
    )
    # Twitter limit is 280 chars — truncate gracefully if needed
    if len(tweet) > 280:
        tweet = tweet[:277] + "..."
    return tweet


def build_text(pitcher_name: str, pitch_type: str, rolling: float,
               baseline: float, drop: float) -> str:
    return (
        f"[DiamondBreakdown Alert]\n"
        f"{pitcher_name} ({pitch_type}): rolling avg {rolling:.1f} mph — "
        f"{drop:.1f} mph below baseline ({baseline:.1f} mph). "
        f"Risk threshold triggered."
    )


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    today      = datetime.today()
    start_date = (today - timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    end_date   = today.strftime("%Y-%m-%d")
    week_str   = today.strftime("%Y-W%W")  # e.g. "2025-W22"

    alert_log  = load_alert_log()
    any_alerts = False

    print(f"\n🔍 DiamondBreakdown Alert Check — {today.strftime('%A %b %d, %Y')}")
    print(f"   Date range: {start_date} → {end_date}")
    print(f"   Threshold:  {RISK_THRESHOLD} mph | Window: {ROLLING_WINDOW} games\n")

    for pitcher in WATCHLIST:
        name       = pitcher["name"]
        pitch_type = pitcher["pitch_type"]

        print(f"Checking {name} ({pitch_type})...")

        df = get_pitcher_data(name, start_date, end_date)
        if df is None:
            print(f"  ⚠ No data returned, skipping.\n")
            continue

        # Also pull full season for a stable baseline
        season_df = get_pitcher_data(name, SEASON_START, end_date)
        if season_df is None:
            season_df = df  # fall back to recent window

        _, season_baseline = compute_rolling_avg(season_df, pitch_type, ROLLING_WINDOW)
        latest_rolling, _  = compute_rolling_avg(df, pitch_type, ROLLING_WINDOW)

        if latest_rolling is None or season_baseline is None:
            print(f"  ⚠ Not enough pitch data for {pitch_type}, skipping.\n")
            continue

        drop = season_baseline - latest_rolling
        print(f"  Rolling avg: {latest_rolling:.1f} mph | Baseline: {season_baseline:.1f} mph | Drop: {drop:.1f} mph")

        if drop >= RISK_THRESHOLD:
            if already_alerted(alert_log, name, week_str):
                print(f"  ℹ Already alerted this week, skipping duplicate.\n")
                continue

            print(f"  🚨 ALERT TRIGGERED — sending tweet + text...")
            tweet_text = build_tweet(name, pitch_type, latest_rolling, season_baseline, drop)
            text_body  = build_text(name, pitch_type, latest_rolling, season_baseline, drop)

            send_tweet(tweet_text)
            send_text(text_body)

            mark_alerted(alert_log, name, week_str)
            any_alerts = True
        else:
            print(f"  ✅ Within threshold, no alert.\n")

    save_alert_log(alert_log)

    if not any_alerts:
        print("\n✅ No alerts fired today.")
    else:
        print("\n🚨 Alerts sent. Check your Twitter and phone.")


if __name__ == "__main__":
    main()
