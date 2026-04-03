# DiamondBreakdown-pyPitchAnalysis
In this repository, I am building and streamlining Python tools to analyze MLB pitchers and how how workload, velocity and movement correlate to injury. The code is organized in Jupyter notebooks to pull MLB Statcast data and create visual representations of the data.  This work supports my YouTube channel, TheDiamondBreakdown.

Project Overview:

Problem:
Major League Baseball is America’s pastime for a reason. Throughout history, generation after generation have passed down the love for the game. However, in recent years, some of the league's premier pitchers have missed playing time. Undoubtedly, significant arm injuries in MLB pitchers has been on the rise. In 2024, there were 240 major pitching injuries, versus 83 in 2010. This increase is alarming, and trying to prevent these injuries before they occur is vital. 

Question:
How do changes in pitching workload, increased velocity, and pitch movement correlate to increased injury risk?
Can we predict future injury using changes found in velocity, pitch count, or pitch movement?

Variables:
Workload (pitch count / innings)
Velocity (average / max)
Spin rate
Movement
Rest Days

Effects: 
Injury occurrence
Injury severity
Injury Risk
Pitching Effectiveness
Longevity

Idea:
Trends in mlb data leading up to injury
Effectiveness before and after injury
Risk of injury and future injury prediction
Pybaseball - python library pulling from baseball savant, baseball reference, and fangraphs
Baseball savant compiles MLB data using MLB Statcast. They develop advanced statistics beyond counting stats. Baseball reference provides a library for all MLB players and their stats. It includes all counting stats, playings, teams, and records. 

https://jupyter.org/ - notebook used to configure and display data
Jupyter can support a variety of coding languages. I have chosen to use Python as it is the language I am most familiar with. Jupyter can create graphs, plots, and other displays, which are easily sharable and exportable to google sheets. 

Visual studio code - development environment 

I used Anthropic's Sonnet 4.6 generative AI model to generate, debug, and optimize the code. 
