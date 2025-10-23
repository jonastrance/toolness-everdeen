# Personal Momentum Map

A lightweight command-line interface for tracking your momentum energy across active projects, habits, and goals. Momentum scores are stored locally in `momentum_data.json`, and the tool provides quick updates, trend visualization, daily recalibration nudges, and an optional "momentum whisperer" insight based on your activity patterns.

## Features

- Momentum score (0–100) per area, adjusted by time decay and weighted updates
- ASCII visualization showing which areas are feeding energy or draining it
- One-line daily recalibration suggestion to rebalance focus
- Quick input syntax for updates: `"ProjectX +3"`, `"Deep Work -1"`, etc.
- Local JSON storage with automatic decay and capped history
- Momentum whisperer agent that highlights weekday momentum spikes

## Installation

The tool depends on Python 3.9+ with no additional third-party packages.

```bash
python3 momentum_map.py --help
```

## Usage

Log quick updates as you move through the day:

```bash
python3 momentum_map.py update "Helionyx +3"
python3 momentum_map.py update "AJ Tickets -2" --note "Blocked by review"
```

View your current momentum map, recalibration suggestion, and whisperer insight:

```bash
python3 momentum_map.py status
```

Reset stored data if you want a fresh start:

```bash
python3 momentum_map.py reset --force
```

The momentum data lives alongside the script in `momentum_data.json`, so you can back it up or sync it however you like.
