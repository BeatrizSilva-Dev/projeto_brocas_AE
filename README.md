# Early Detection of Drill Jamming using Ultrasonic Signals

This repository contains the source code, datasets, and experimental results for the research on early detection of drill jamming in CNC machining processes using Machine Learning (Autoencoders and XGBoost).

## Project Overview
The study focuses on monitoring 4 mm carbide drills throughout their entire life cycle. By analyzing acoustic signals in the ultrasonic range, we propose a modular system capable of identifying subtle spectral shifts that precede catastrophic failure (jamming).

## Repository Structure

The project is organized as follows:

```text
.
├── data/
│   ├── raw/             # Original ultrasonic recordings (.wav)
│   ├── segmented/       # Audio segments processed for training
│   └── results/         # Output CSV files from models 
├── plots/               # High-resolution figures used in the paper
├── src/
│   ├── preprocessing/   # Scripts for audio 
│   ├── optimization/    # Model architectures and hyperparameter tuning
│   └── analysis/        # Comparative plots
├── .gitignore           
├── requirements.txt     
└── README.md           
