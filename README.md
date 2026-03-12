# Adaptive Evolutionary Traffic Optimization 🚦

A custom Evolutionary Algorithm (EA) built to optimize traffic light timings across a continuous, bidirectional 4x4 urban grid. Developed as a mini-project for CISC455/851.

## 🎥 Simulation Demo

*(Ensure `demo.mp4` is in the root directory of your repository for this to render correctly on GitHub)*

<video src="demo.mp4" controls="controls" style="max-width: 100%; height: auto;">
  Your browser does not support the video tag.
</video>

## 📋 Project Overview

This project simulates a highly complex traffic environment to test the efficacy of Evolutionary Algorithms in solving steady-state optimization problems. Instead of relying on human-designed fixed traffic cycles, this engine evolves its own timing strategies by simulating thousands of traffic scenarios, selecting the most efficient sequences, and genetically combining them over multiple generations.

### Key Features
* **Custom 4x4 Grid Physics Engine:** Simulates bidirectional traffic (North, South, East, West) with variable intersection distances.
* **Realistic Vehicle Dynamics:** Differentiates between vehicle types (cars vs. buses) with variable lengths and acceleration rates, and strictly enforces a 2m safety buffer.
* **Continuous Traffic Pressure:** Vehicles spawn continuously (steady-state) rather than in a single wave, forcing the EA to manage ongoing congestion and preventing the "ghost town" effect.
* **Headless Training Mode:** Bypasses the Pygame rendering engine for high-speed, CPU-focused algorithm training.

## 🧬 Evolutionary Algorithm Design

Our EA is designed to find the optimal balance between exploration and exploitation using the following mechanics:

* **Genotype (Representation):** Traffic timings are represented as a sequence of `TimingBlock` objects. Each block strictly enforces mandatory safety phases (6s Yellow, 3s All-Red) to ensure the EA cannot "cheat" by creating dangerous, illegal light transitions.
* **Fitness Function (Minimization):** The algorithm scores strategies based on `Travel Time + (Idling Time * 2)`. Unfinished vehicles apply a distance-based penalty gradient, allowing the EA to learn even in early generations where zero vehicles finish the route.
* **Selection:** Utilizes **Tournament Selection** ($k=3$) to apply positive selection pressure without prematurely converging on local optima.
* **Variation (Crossover):** Uses **Block-Level Crossover** to swap complete light cycles between two parents, ensuring the mandatory safety constraints are never spliced or corrupted.
* **Mutation:** Applies a 30% chance to randomly re-initialize a specific timing block, injecting new genetic material to explore alternative timing combinations.
* **Elitism:** The absolute best-performing individual of each generation is guaranteed survival into the next generation.

## 📁 Repository Structure

* `StraightLineTraining.ipynb` - The core engine. Handles the evolutionary loop, physics simulation, and headless training. Outputs the trained model.
* `Validation.ipynb` - The analysis suite. Loads the trained model to render a visual replay and generates matplotlib charts (e.g., Gridlock Density) for performance analysis.
* `best_timing.pkl` - A serialized artifact containing the genotype of the highest-performing traffic strategy found during training.

## 🚀 Getting Started

### Prerequisites
You will need Python 3.x and the following libraries installed:
```bash
pip install numpy pygame matplotlib jupyter