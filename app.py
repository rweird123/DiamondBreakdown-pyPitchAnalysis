import streamlit as st
from pybaseball import statcast_pitcher, playerid_lookup
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
import requests

st.set_page_config(page_title="Diamond Breakdown | Pitcher Analytics", layout="wide")

# ── Pitcher Roster (autocomplete source) ─────────────────────────────────────
@st.cache_data(show_spinner="Loading pitcher roster...")
def load_pitcher_roster():
    """Pull the full pybaseball people table and return a sorted list of
    'First Last' strings for every player who has a known MLB pitching role.
    Falls back to an empty list if the network call fails."""
    try:
        from pybaseball import chadwick_register
        people = chadwick_register(save=False)
        # keep rows that have a usable name
        people = people.dropna(subset=["name_first", "name_last", "key_mlbam"])
        people["full_name"] = (
            people["name_first"].str.strip().str.title()
            + " "
            + people["name_last"].str.strip().str.title()
        )
        names = sorted(people["full_name"].dropna().unique().tolist())
        return names
    except Exception:
        return []

# ── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@400;600;700&family=Barlow:wght@400;500&display=swap');

  html, body, [class*="css"] { font-family: 'Barlow', sans-serif; }

  h1, h2, h3 { font-family: 'Barlow Condensed', sans-serif !important; letter-spacing: 0.04em; }

  [data-testid="stMetricValue"] { font-family: 'Barlow Condensed', sans-serif; font-size: 2rem !important; color: #4FC3F7; }
  [data-testid="stMetricLabel"] { font-size: 0.75rem !important; color: #888 !important; text-transform: uppercase; letter-spacing: 0.08em; }

  .risk-box { background: #1a0a0a; border: 1px solid #7f1d1d; border-radius: 8px; padding: 1rem 1.2rem; margin-top: 0.5rem; }
  .ai-box   { background: #0a0f1a; border: 1px solid #1e3a5f; border-radius: 8px; padding: 1.2rem 1.4rem; margin-top: 0.5rem; line-height: 1.7; }
  .section-label { font-family: 'Barlow Condensed', sans-serif; font-size: 0.7rem; letter-spacing: 0.15em; color: #4FC3F7; text-transform: uppercase; margin-bottom: 0.25rem; }
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("# ⚾ Diamond Breakdown")
st.markdown("**Pitcher Intelligence Platform** — Velocity · Spin · Movement · AI Risk Assessment")
st.divider()

# ── Constants ────────────────────────────────────────────────────────────────
PITCH_NAME_MAP = {
    "FF": "Four-Seam Fastball", "SI": "Sinker", "SL": "Slider",
    "CH": "Changeup", "CU": "Curveball", "FC": "Cutter",
    "KC": "Knuckle Curve", "FS": "Splitter", "FO": "Forkball",
    "EP": "Eephus", "KN": "Knuckleball", "SC": "Screwball",
}
PITCH_CODE_MAP = {v: k for k, v in PITCH_NAME_MAP.items()}
COLORS = ["#4FC3F7", "#FF8A65", "#A5D6A7", "#CE93D8", "#FFF176", "#80DEEA"]

DARK_BG   = "#0e1117"
PANEL_BG  = "#161b24"
GRID_CLR  = "#1e2533"
TEXT_CLR  = "#c9d1d9"
AXIS_CLR  = "#4a5568"

def apply_dark_style(ax):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=TEXT_CLR, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(AXIS_CLR)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, color=GRID_CLR, linewidth=0.5)
    ax.xaxis.label.set_color(TEXT_CLR)
    ax.yaxis.label.set_color(TEXT_CLR)
    ax.title.set_color("white")

# ── Pitcher Roster Load ───────────────────────────────────────────────────────
pitcher_roster = load_pitcher_roster()
# Ensure a handful of popular names are always present even if roster fails
_fallbacks = [
    "Corbin Burnes", "Gerrit Cole", "Zack Wheeler", "Spencer Strider",
    "Logan Webb", "Tarik Skubal", "Hunter Brown", "Freddy Peralta",
    "Chris Sale", "Kevin Gausman", "Luis Castillo", "Max Fried",
    "Framber Valdez", "Dylan Cease", "Sandy Alcantara", "Blake Snell",
    "Justin Verlander", "Clayton Kershaw", "Tyler Glasnow", "Justin Steele",
]
if not pitcher_roster:
    pitcher_roster = sorted(_fallbacks)
else:
    # merge fallbacks in case they're missing
    pitcher_set = set(pitcher_roster)
    for f in _fallbacks:
        if f not in pitcher_set:
            pitcher_roster.append(f)
    pitcher_roster = sorted(pitcher_roster)

def pitcher_selectbox(label, default, key):
    idx = pitcher_roster.index(default) if default in pitcher_roster else 0
    return st.selectbox(label, pitcher_roster, index=idx, key=key,
                        help="Start typing to search")

# ── Mode Toggle ──────────────────────────────────────────────────────────────
mode = st.radio("Mode", ["Single Pitcher", "Compare Two Pitchers"], horizontal=True)
compare_mode = mode == "Compare Two Pitchers"

# ── Season / Period Picker ────────────────────────────────────────────────────
st.markdown("**Select Season & Period**")
_season_col, _period_col = st.columns(2)

with _season_col:
    selected_season = st.selectbox(
        "Season",
        options=sorted(MLB_SEASONS.keys(), reverse=True),
        index=0,
        key="season_pick",
    )

_valid_periods = get_valid_periods(selected_season)

if not _valid_periods:
    st.warning("No data available yet for that season.")
    st.stop()

with _period_col:
    selected_period = st.selectbox(
        "Period",
        options=list(_valid_periods.keys()),
        key="period_pick",
    )

start_date, end_date = _valid_periods[selected_period]
start_date = pd.to_datetime(start_date).date()
end_date   = pd.to_datetime(end_date).date()

st.caption(
    f"Date range: **{start_date.strftime('%b %d, %Y')}** to **{end_date.strftime('%b %d, %Y')}**"
    + (" *(season in progress — data through today)*"
       if end_date == _dt.date.today() else "")
)

# ── Inputs ───────────────────────────────────────────────────────────────────
if compare_mode:
    c1, c2 = st.columns(2)
    with c1: player_name  = pitcher_selectbox("Pitcher 1", "Corbin Burnes", "p1")
    with c2: player_name2 = pitcher_selectbox("Pitcher 2", "Gerrit Cole",   "p2")
else:
    player_name = pitcher_selectbox("Pitcher Name", "Corbin Burnes", "p1")

c4, c5 = st.columns(2)
with c4: rolling_window  = st.slider("Rolling Average Window (games)", 3, 10, 5)
with c5: risk_threshold  = st.slider("Risk Flag Threshold (mph drop)", 0.5, 3.0, 1.5, step=0.1)

selected_pitches = st.multiselect(
    "Pitch Types to Display",
    list(PITCH_NAME_MAP.values()),
    default=["Four-Seam Fastball", "Slider"],
)

enable_ai = st.toggle("🤖 Enable AI Injury Risk Summary", value=True)

# ── Data Fetching ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Fetching Statcast data...")
def get_pitcher_data(name: str, start: str, end: str):
    parts = name.strip().split()
    if len(parts) < 2:
        return None, None
    first, last = parts[0], parts[-1]
    lookup = playerid_lookup(last, first)
    if lookup.empty:
        return None, None
    # If multiple rows, prefer the one with the most recent mlb activity
    if len(lookup) > 1 and "mlb_played_last" in lookup.columns:
        lookup = lookup.sort_values("mlb_played_last", ascending=False)
    pid  = lookup["key_mlbam"].iloc[0]
    data = statcast_pitcher(start, end, pid)
    return data, pid

# ── AI Risk Summary ───────────────────────────────────────────────────────────
def generate_ai_summary(pitcher_name: str, risk_lines: list, baseline_info: dict, pitch_stats: dict) -> str:
    stats_text = "\n".join([
        f"- {p}: avg velo {v['velo']:.1f} mph, avg spin {v['spin']:.0f} rpm, "
        f"H-break {v['hbreak']:.1f} in, V-break {v['vbreak']:.1f} in"
        for p, v in pitch_stats.items()
    ])
    risk_text = "\n".join(risk_lines) if risk_lines else "No risk zones flagged."
    baseline_text = "\n".join([f"- {p}: {b:.1f} mph baseline" for p, b in baseline_info.items()])

    prompt = f"""You are a baseball analytics scout writing a concise injury risk assessment for {pitcher_name}.

Pitch baselines:
{baseline_text}

Pitch statistics:
{stats_text}

Risk zones detected:
{risk_text}

Write a 3–4 sentence scouting-style injury risk summary. Be direct, analytical, and specific. 
Mention actual numbers. Flag any concerning patterns. If no risk zones were found, note that the pitcher 
looks healthy and consistent. End with a one-sentence overall risk verdict (Low / Moderate / High risk).
Do not use bullet points — write in flowing prose like a real scouting report."""

    try:
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {st.secrets['GROQ_API_KEY']}",
            },
            json={
                "model": "llama-3.3-70b-versatile",
                "max_tokens": 1000,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=30,
        )
        data = resp.json()
        if "error" in data:
            return f"API Error: {data['error']['message']}"
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"AI summary unavailable: {e}"

# ── Velocity Chart (single or comparison) ────────────────────────────────────
def build_velo_chart(pitcher_df, player_name, start_date, end_date, selected_pitches,
                     rolling_window, risk_threshold, color_offset=0, ax=None):
    full_df = pd.DataFrame({"game_date": pd.date_range(start=start_date, end=end_date)})
    risk_report  = []
    baseline_info = {}
    pitch_stats   = {}

    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 5))
        fig.patch.set_facecolor(DARK_BG)
        standalone = True
    else:
        standalone = False

    apply_dark_style(ax)

    for i, pitch_full in enumerate(selected_pitches):
        pitch_code = PITCH_CODE_MAP[pitch_full]
        pitch_df   = pitcher_df[pitcher_df["pitch_type"] == pitch_code].copy()
        if pitch_df.empty:
            continue

        pitch_df["velo"] = pd.to_numeric(pitch_df["release_speed"], errors="coerce")
        pitch_df = pitch_df.dropna(subset=["velo"])

        # Collect advanced stats
        spin   = pd.to_numeric(pitch_df.get("release_spin_rate", pd.Series()), errors="coerce").mean()
        hbreak = pd.to_numeric(pitch_df.get("pfx_x", pd.Series()), errors="coerce").mean() * 12
        vbreak = pd.to_numeric(pitch_df.get("pfx_z", pd.Series()), errors="coerce").mean() * 12
        pitch_stats[pitch_full] = {
            "velo": pitch_df["velo"].mean(),
            "spin": spin if not np.isnan(spin) else 0,
            "hbreak": hbreak if not np.isnan(hbreak) else 0,
            "vbreak": vbreak if not np.isnan(vbreak) else 0,
        }

        game_avg = pitch_df.groupby("game_date")["velo"].mean().reset_index()
        game_avg = pd.merge(full_df, game_avg, on="game_date", how="left")
        game_avg["velo_interp"] = game_avg["velo"].interpolate()
        game_avg["velo_rolling"] = game_avg["velo_interp"].rolling(rolling_window, min_periods=1).mean()

        color           = COLORS[(i + color_offset) % len(COLORS)]
        season_baseline = game_avg["velo"].median()
        baseline_info[pitch_full] = season_baseline

        ax.plot(game_avg["game_date"], game_avg["velo"], alpha=0.15, linestyle="--", color=color)
        ax.plot(game_avg["game_date"], game_avg["velo_rolling"],
                label=f"{pitch_full} (rolling avg)", color=color, linewidth=2)
        ax.axhline(season_baseline, linestyle=":", linewidth=1, color=color, alpha=0.5,
                   label=f"{pitch_full} baseline ({season_baseline:.1f} mph)")

        flagged = game_avg.dropna(subset=["velo"]).copy()
        flagged["drop"] = season_baseline - flagged["velo_rolling"]
        risk_games = flagged[flagged["drop"] >= risk_threshold]

        if not risk_games.empty:
            for _, row in risk_games.iterrows():
                ax.axvspan(row["game_date"] - pd.Timedelta(days=2),
                           row["game_date"] + pd.Timedelta(days=2),
                           color="red", alpha=0.10)
            ax.scatter(risk_games["game_date"], risk_games["velo_rolling"],
                       color="red", zorder=5, s=55)
            risk_report.append(f"**{pitch_full}**")
            for _, row in risk_games.iterrows():
                risk_report.append(
                    f"  • {row['game_date'].strftime('%b %d')}: "
                    f"{row['velo_rolling']:.1f} mph — **{row['drop']:.1f} mph below baseline** ⚠️"
                )

    ax.set_title(f"{player_name} — Velocity Trends", fontsize=12, pad=8)
    ax.set_xlabel("Date"); ax.set_ylabel("Velocity (mph)")

    handles, _ = ax.get_legend_handles_labels()
    red_patch   = mpatches.Patch(color="red", alpha=0.4, label=f"Risk zone (≥{risk_threshold} mph drop)")
    ax.legend(handles=handles + [red_patch], facecolor="#1a1a2e", edgecolor=AXIS_CLR,
              labelcolor="white", fontsize=7)

    if standalone:
        return fig, risk_report, baseline_info, pitch_stats
    return risk_report, baseline_info, pitch_stats

# ── Advanced Analytics Chart ──────────────────────────────────────────────────
def build_analytics_charts(pitcher_df, player_name, selected_pitches):
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor(DARK_BG)
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    ax_spin   = fig.add_subplot(gs[0, 0])
    ax_hbreak = fig.add_subplot(gs[0, 1])
    ax_vbreak = fig.add_subplot(gs[0, 2])
    ax_scatter= fig.add_subplot(gs[1, 0:2])
    ax_usage  = fig.add_subplot(gs[1, 2])

    for ax in [ax_spin, ax_hbreak, ax_vbreak, ax_scatter, ax_usage]:
        apply_dark_style(ax)

    pitch_labels, spin_vals, hbreak_vals, vbreak_vals = [], [], [], []

    for i, pitch_full in enumerate(selected_pitches):
        code     = PITCH_CODE_MAP[pitch_full]
        pitch_df = pitcher_df[pitcher_df["pitch_type"] == code].copy()
        if pitch_df.empty:
            continue

        color = COLORS[i % len(COLORS)]
        pitch_df["game_date"] = pd.to_datetime(pitch_df["game_date"])
        game_spin = (
            pitch_df.assign(spin=pd.to_numeric(pitch_df["release_spin_rate"], errors="coerce"))
            .dropna(subset=["spin"]).groupby("game_date")["spin"].mean().reset_index()
        )

        if not game_spin.empty:
            ax_spin.plot(game_spin["game_date"], game_spin["spin"], color=color, linewidth=1.8,
                         label=pitch_full, alpha=0.9)

        # Movement scatter (per pitch)
        hx = pd.to_numeric(pitch_df.get("pfx_x", pd.Series(dtype=float)), errors="coerce") * 12
        hz = pd.to_numeric(pitch_df.get("pfx_z", pd.Series(dtype=float)), errors="coerce") * 12
        valid = (~hx.isna()) & (~hz.isna())
        if valid.sum() > 0:
            ax_scatter.scatter(hx[valid], hz[valid], color=color, alpha=0.25, s=12, label=pitch_full)
            # Centroid
            ax_scatter.scatter(hx[valid].mean(), hz[valid].mean(), color=color,
                               s=120, edgecolors="white", linewidths=1.2, zorder=10)

        # Bar data
        pitch_labels.append(pitch_full.replace(" ", "\n"))
        hbreak_vals.append(hx.mean() if valid.sum() > 0 else 0)
        vbreak_vals.append(hz.mean() if valid.sum() > 0 else 0)

    # Spin rate over time
    ax_spin.set_title("Spin Rate Over Time", fontsize=10)
    ax_spin.set_xlabel("Date"); ax_spin.set_ylabel("Spin Rate (rpm)")
    ax_spin.legend(facecolor="#1a1a2e", edgecolor=AXIS_CLR, labelcolor="white", fontsize=7)
    ax_spin.tick_params(axis='x', rotation=30)

    # H-Break bar
    x_pos   = np.arange(len(pitch_labels))
    bar_clrs = [COLORS[i % len(COLORS)] for i in range(len(pitch_labels))]
    ax_hbreak.bar(x_pos, hbreak_vals, color=bar_clrs, alpha=0.85, width=0.5)
    ax_hbreak.set_xticks(x_pos); ax_hbreak.set_xticklabels(pitch_labels, fontsize=7)
    ax_hbreak.set_title("Horizontal Break (in)", fontsize=10)
    ax_hbreak.set_ylabel("Inches"); ax_hbreak.axhline(0, color=AXIS_CLR, linewidth=0.8)

    # V-Break bar
    ax_vbreak.bar(x_pos, vbreak_vals, color=bar_clrs, alpha=0.85, width=0.5)
    ax_vbreak.set_xticks(x_pos); ax_vbreak.set_xticklabels(pitch_labels, fontsize=7)
    ax_vbreak.set_title("Vertical Break / Rise (in)", fontsize=10)
    ax_vbreak.set_ylabel("Inches"); ax_vbreak.axhline(0, color=AXIS_CLR, linewidth=0.8)

    # Movement scatter
    ax_scatter.axhline(0, color=AXIS_CLR, linewidth=0.6)
    ax_scatter.axvline(0, color=AXIS_CLR, linewidth=0.6)
    ax_scatter.set_title("Pitch Movement Profile (pitcher's POV)", fontsize=10)
    ax_scatter.set_xlabel("Horizontal Break (in)"); ax_scatter.set_ylabel("Vertical Break (in)")
    ax_scatter.legend(facecolor="#1a1a2e", edgecolor=AXIS_CLR, labelcolor="white", fontsize=7)

    # Pitch usage pie
    usage_counts = {}
    for pitch_full in selected_pitches:
        code  = PITCH_CODE_MAP[pitch_full]
        count = (pitcher_df["pitch_type"] == code).sum()
        if count > 0:
            usage_counts[pitch_full] = count

    if usage_counts:
        pie_labels = [k.replace(" ", "\n") for k in usage_counts.keys()]
        pie_colors = [COLORS[i % len(COLORS)] for i in range(len(usage_counts))]
        wedges, texts, autotexts = ax_usage.pie(
            list(usage_counts.values()), labels=pie_labels, autopct="%1.0f%%",
            colors=pie_colors, textprops={"color": TEXT_CLR, "fontsize": 7},
            wedgeprops={"linewidth": 0.5, "edgecolor": DARK_BG},
        )
        for at in autotexts:
            at.set_color("white"); at.set_fontsize(7)
        ax_usage.set_title("Pitch Usage Mix", fontsize=10)

    fig.suptitle(f"{player_name} — Advanced Analytics", color="white", fontsize=13,
                 fontfamily="monospace", y=0.98)
    return fig

# ── Comparison Chart ──────────────────────────────────────────────────────────
def build_comparison_chart(df1, df2, name1, name2, selected_pitches, rolling_window, risk_threshold, start_date, end_date):
    fig, axes = plt.subplots(len(selected_pitches), 1,
                             figsize=(14, 4 * max(len(selected_pitches), 1)),
                             squeeze=False)
    fig.patch.set_facecolor(DARK_BG)
    full_df = pd.DataFrame({"game_date": pd.date_range(start=start_date, end=end_date)})

    for row_i, pitch_full in enumerate(selected_pitches):
        ax    = axes[row_i][0]
        code  = PITCH_CODE_MAP[pitch_full]
        apply_dark_style(ax)

        for p_i, (df, name, lc) in enumerate([(df1, name1, COLORS[0]), (df2, name2, COLORS[1])]):
            p_df = df[df["pitch_type"] == code].copy()
            if p_df.empty:
                continue
            p_df["velo"] = pd.to_numeric(p_df["release_speed"], errors="coerce")
            p_df = p_df.dropna(subset=["velo"])
            game_avg = p_df.groupby("game_date")["velo"].mean().reset_index()
            game_avg = pd.merge(full_df, game_avg, on="game_date", how="left")
            game_avg["velo_interp"]  = game_avg["velo"].interpolate()
            game_avg["velo_rolling"] = game_avg["velo_interp"].rolling(rolling_window, min_periods=1).mean()
            baseline = game_avg["velo"].median()

            ax.plot(game_avg["game_date"], game_avg["velo_rolling"],
                    label=f"{name} ({baseline:.1f} mph avg)", color=lc, linewidth=2)
            ax.axhline(baseline, linestyle=":", color=lc, alpha=0.5, linewidth=1)

        ax.set_title(f"{pitch_full} Velocity Comparison", fontsize=10)
        ax.set_ylabel("Velocity (mph)")
        ax.legend(facecolor="#1a1a2e", edgecolor=AXIS_CLR, labelcolor="white", fontsize=8)

    axes[-1][0].set_xlabel("Date")
    fig.suptitle(f"{name1}  vs  {name2}", color="white", fontsize=14, y=1.01)
    fig.tight_layout()
    return fig

# ── Summary Metrics ───────────────────────────────────────────────────────────
def show_summary_metrics(pitcher_df, selected_pitches):
    cols = st.columns(len(selected_pitches))
    for i, pitch_full in enumerate(selected_pitches):
        code = PITCH_CODE_MAP[pitch_full]
        pf   = pitcher_df[pitcher_df["pitch_type"] == code].copy()
        if pf.empty:
            continue
        velo = pd.to_numeric(pf["release_speed"], errors="coerce")
        spin = pd.to_numeric(pf.get("release_spin_rate", pd.Series(dtype=float)), errors="coerce")
        with cols[i]:
            st.metric(f"{pitch_full}", f"{velo.mean():.1f} mph", f"σ {velo.std():.2f}")
            st.caption(f"Spin: {spin.mean():.0f} rpm  |  n={len(pf):,} pitches")

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
if st.button("⚡ Generate Dashboard", type="primary"):

    # ── Fetch pitcher 1 ──────────────────────────────────────────────────────
    with st.spinner(f"Loading {player_name}..."):
        pitcher_df, player_id = get_pitcher_data(player_name, str(start_date), str(end_date))

    if pitcher_df is None or pitcher_df.empty:
        st.error(f"Couldn't find data for **{player_name}**. Check spelling.")
        st.stop()

    pitcher_df["game_date"] = pd.to_datetime(pitcher_df["game_date"])

    # ── Fetch pitcher 2 (compare mode) ───────────────────────────────────────
    if compare_mode:
        with st.spinner(f"Loading {player_name2}..."):
            pitcher_df2, _ = get_pitcher_data(player_name2, str(start_date), str(end_date))
        if pitcher_df2 is None or pitcher_df2.empty:
            st.error(f"Couldn't find data for **{player_name2}**. Check spelling.")
            st.stop()
        pitcher_df2["game_date"] = pd.to_datetime(pitcher_df2["game_date"])

    # ═════════════════════════════════════════════════════════════════════════
    # TABS
    # ═════════════════════════════════════════════════════════════════════════
    if compare_mode:
        tabs = st.tabs(["🆚 Comparison", "📊 P1 Analytics", "📊 P2 Analytics", "🤖 AI Report"])
    else:
        tabs = st.tabs(["📈 Velocity", "📊 Advanced Analytics", "🤖 AI Report"])

    # ── TAB: Velocity / Comparison ────────────────────────────────────────────
    with tabs[0]:
        if compare_mode:
            st.subheader(f"{player_name}  vs  {player_name2}")
            comp_fig = build_comparison_chart(
                pitcher_df, pitcher_df2, player_name, player_name2,
                selected_pitches, rolling_window, risk_threshold, start_date, end_date
            )
            st.pyplot(comp_fig)

            st.divider()
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**{player_name}** summary")
                show_summary_metrics(pitcher_df, selected_pitches)
            with c2:
                st.markdown(f"**{player_name2}** summary")
                show_summary_metrics(pitcher_df2, selected_pitches)

        else:
            # Single pitcher velocity tab
            show_summary_metrics(pitcher_df, selected_pitches)
            st.divider()
            velo_fig, risk_report, baseline_info, pitch_stats = build_velo_chart(
                pitcher_df, player_name, start_date, end_date,
                selected_pitches, rolling_window, risk_threshold,
            )
            st.pyplot(velo_fig)

            st.divider()
            st.subheader("🚨 Risk Report")
            if risk_report:
                st.markdown(f"Games where rolling avg dropped **≥ {risk_threshold} mph** below season baseline:")
                st.markdown('<div class="risk-box">', unsafe_allow_html=True)
                for line in risk_report:
                    st.markdown(line)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.success(f"✅ No risk zones detected. Velocity stayed within {risk_threshold} mph of baseline.")

    # ── TAB: Advanced Analytics (P1) ─────────────────────────────────────────
    analytics_tab_idx = 1
    with tabs[analytics_tab_idx]:
        st.subheader(f"{player_name} — Advanced Analytics")
        analytics_fig = build_analytics_charts(pitcher_df, player_name, selected_pitches)
        st.pyplot(analytics_fig)

    # ── TAB: Advanced Analytics (P2, compare mode only) ──────────────────────
    if compare_mode:
        with tabs[2]:
            st.subheader(f"{player_name2} — Advanced Analytics")
            analytics_fig2 = build_analytics_charts(pitcher_df2, player_name2, selected_pitches)
            st.pyplot(analytics_fig2)

    # ── TAB: AI Report ────────────────────────────────────────────────────────
    ai_tab_idx = 3 if compare_mode else 2
    with tabs[ai_tab_idx]:
        st.subheader("🤖 AI Injury Risk Assessment")

        if not enable_ai:
            st.info("Enable the AI toggle above to generate this report.")
        else:
            # Build context for AI (always from P1; comparison gives both)
            _, risk_report_ai, baseline_info_ai, pitch_stats_ai = build_velo_chart(
                pitcher_df, player_name, start_date, end_date,
                selected_pitches, rolling_window, risk_threshold,
            )

            with st.spinner("Generating scouting report..."):
                summary = generate_ai_summary(player_name, risk_report_ai, baseline_info_ai, pitch_stats_ai)

            st.markdown(f'<div class="section-label">Scouting Report — {player_name}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="ai-box">{summary}</div>', unsafe_allow_html=True)

            if compare_mode:
                _, risk_report_ai2, baseline_info_ai2, pitch_stats_ai2 = build_velo_chart(
                    pitcher_df2, player_name2, start_date, end_date,
                    selected_pitches, rolling_window, risk_threshold,
                )
                with st.spinner(f"Generating report for {player_name2}..."):
                    summary2 = generate_ai_summary(player_name2, risk_report_ai2, baseline_info_ai2, pitch_stats_ai2)

                st.divider()
                st.markdown(f'<div class="section-label">Scouting Report — {player_name2}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="ai-box">{summary2}</div>', unsafe_allow_html=True)

            # Export
            st.divider()
            export_lines = [
                f"# Diamond Breakdown — AI Risk Report",
                f"**{player_name}**\n",
                summary,
            ]
            if compare_mode:
                export_lines += [f"\n---\n**{player_name2}**\n", summary2]
            st.download_button(
                "⬇️ Download Report (.txt)",
                data="\n".join(export_lines),
                file_name="diamond_breakdown_report.txt",
                mime="text/plain",
            )
