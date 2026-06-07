# DiamondBreakdown-pyPitchAnalysis

**[Live Streamlit App](https://diamondbreakdown-pypitchanalysis.streamlit.app/)** · **[DiamondBreakdown YouTube Channel](https://www.youtube.com/@TheDiamondBreakdown)**



## Project Overview:

A data analysis project investigating the rise of pitcher arm injuries in MLB, using Statcast advanced data to identify correlations between pitcher workload, velocity and movement correlate to injury. The code is organized in Jupyter notebooks to pull MLB Statcast data and create visual representations of the data, which was combined into a working Streamlit web app. This work also supports my YouTube channel, TheDiamondBreakdown.

## Problem

Major League Baseball is America’s pastime for a reason. Throughout history, generation after generation have passed down the love for the game. However, in recent years, some of the league's premier pitchers have missed playing time. Undoubtedly, significant arm injuries in MLB pitchers has been on the rise. In 2024, there were 240 major pitching injuries, versus 83 in 2010. This increase is alarming, and trying to prevent these injuries before they occur is vital. 

## Question

How do changes in pitching workload, increased velocity, and pitch movement correlate to increased injury risk?
Can we predict future injury using changes found in velocity, pitch count, or pitch movement?

## Variables

- Workload (pitch count / innings)
- Velocity (average / max)
- Spin rate
- Movement
- Rest Days

## Effects

- Injury occurrence
- Injury severity
- Injury Risk
- Pitching Effectiveness
- Longevity

## Idea

- Trends in MLB data leading up to injury
- Effectiveness before and after injury
- Risk of injury and future injury prediction


## Tools

- Pybaseball - python library pulling from baseball savant, baseball reference, and fangraphs. Baseball savant compiles MLB data using MLB Statcast. They develop advanced statistics beyond counting stats. Baseball reference provides a library for all MLB players and their stats. It includes all counting stats, playings, teams, and records. 

- [Jupyter](https://jupyter.org/) - notebook used to configure and display data
Jupyter can support a variety of coding languages. I have chosen to use Python as it is the language I am most familiar with. Jupyter can create graphs, plots, and other displays, which are easily sharable and exportable to google sheets. 

- Visual studio code - development environment 

- Claude Sonnet 4.6 - Generative AI model owned by Anthropic, it can be used for daily tasks, complex problem solving, and code. I will use this to assist me in learning, writing, and debugging code. I will not explicitly copy full solutions from AI, but I will instead use it to explain errors, suggest improvements, and help me better understand how to use various software. I have very basic coding skills, and this project was not necessarily created to work on coding, but instead to focus on data analysis and MLB pitchers. 

All of the writing on this document, as well as on my YouTube channel and the GitHub repository are fully created by myself, Rohan Shah. 

## App - [Pitcher Intelligence Dashboard](https://diamondbreakdown-pypitchanalysis.streamlit.app/)

A Streamlit web app created for analyzing MLB pitchers

**Features**
- Search any MLB pitcher by name (full league autocomplete)
- View velocity trend charts with rolling averages and season baselines
- Injury risk flagging - highlights games where velocity drops below a specified value
- IL stint overlays on velocity trend charts pulled from the MLB Stats API
- Advanced analytic charts: spin rate over time, horizontal/vertical break, pitch movement profiles, pitch usage
- Side-by-side pitcher comparison mode
- AI-generated scouting reports and injury risk identifier (Groq / Llama 3.3)
- Season and period selection (Spring Training, Regular Season, Postseason)
