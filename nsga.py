import pandas as pd
import numpy as np
import random
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt
import plotly.express as px
from umap import UMAP
import os

# Load data from /data/ folder
data_dir = "./data"
drivers = pd.read_csv(os.path.join(data_dir, "DRIVERS.csv"))
gliders = pd.read_csv(os.path.join(data_dir, "GLIDERS.csv"))
tires = pd.read_csv(os.path.join(data_dir, "TIRES.csv"))
vehicles = pd.read_csv(os.path.join(data_dir, "VEHICLES.csv"))

# Performance metrics
stat_cols = ['GroundSpeed', 'WaterSpeed', 'AirSpeed', 'AntiGravitySpeed',
             'Acceleration', 'Weight', 'GroundHandling', 'WaterHandling',
             'AirHandling', 'AntiGravityHandling', 'Traction', 'MiniTurbo']

# Combine stats for a setup (driver, vehicle, tires, glider)
def combine_stats(ids):
    d, v, t, g = ids
    parts = [drivers.iloc[d], vehicles.iloc[v], tires.iloc[t], gliders.iloc[g]]
    total = sum([p[stat_cols].values for p in parts])
    return total.tolist()

# Setup NSGA-II
creator.create("FitnessMulti", base.Fitness, weights=(1.0,) * len(stat_cols))
creator.create("Individual", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()

toolbox.register("driver_id", random.randint, 0, len(drivers) - 1)
toolbox.register("vehicle_id", random.randint, 0, len(vehicles) - 1)
toolbox.register("tire_id", random.randint, 0, len(tires) - 1)
toolbox.register("glider_id", random.randint, 0, len(gliders) - 1)

toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.driver_id, toolbox.vehicle_id, toolbox.tire_id, toolbox.glider_id), n=1)

# Register the population
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evaluate(individual):
    return combine_stats(individual)

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
def custom_mutate(individual, indpb):
    if random.random() < indpb:
        individual[0] = random.randint(0, len(drivers) - 1)
    if random.random() < indpb:
        individual[1] = random.randint(0, len(vehicles) - 1)
    if random.random() < indpb:
        individual[2] = random.randint(0, len(tires) - 1)
    if random.random() < indpb:
        individual[3] = random.randint(0, len(gliders) - 1)
    return individual,
toolbox.register("mutate", custom_mutate, indpb=0.2)
toolbox.register("select", tools.selNSGA2)

# Run NSGA-II optimization
pop = toolbox.population(n=100)
hof = tools.ParetoFront()
algorithms.eaMuPlusLambda(pop, toolbox, mu=100, lambda_=200, cxpb=0.6, mutpb=0.3, ngen=40, halloffame=hof, verbose=True)

# Convert to DataFrame
records = []
for ind in hof:
    d, v, t, g = ind
    stats = combine_stats(ind)
    record = {
        **dict(zip(stat_cols, stats)),
        "Driver": drivers.iloc[d]["Name"],
        "Vehicle": vehicles.iloc[v]["Name"],
        "Tires": tires.iloc[t]["Name"],
        "Glider": gliders.iloc[g]["Name"]
    }
    records.append(record)

optimized_df = pd.DataFrame(records)

# Score setups based on track type
def score_setups(df, weights):
    scores = []
    for _, row in df.iterrows():
        score = 0
        for stat, weight in weights.items():
            score += weight * row[stat]
        scores.append(score)
    return scores

# Define stat weights for each track type
track_weights = {
    "long_straights": {
        "GroundSpeed": 1.5,
        "AirSpeed": 1.2,
        "Weight": 1.0,
        "Acceleration": 0.5,
        "MiniTurbo": 0.3,
        "Traction": 0.2
    },
    "curvy": {
        "Acceleration": 1.2,
        "MiniTurbo": 1.5,
        "Traction": 1.2,
        "GroundHandling": 1.0,
        "AirHandling": 1.0,
        "AntiGravityHandling": 1.0,
        "Weight": 0.5
    },
    "offroading": {
        "Traction": 1.5,
        "GroundHandling": 1.2,
        "WaterHandling": 1.0,
        "AirHandling": 1.0,
        "AntiGravityHandling": 1.0,
        "Acceleration": 1.5,
        "Weight": 1.0,
        "MiniTurbo": 1.0
    }
}

# Compute scores and add as new columns
for track_type, weights in track_weights.items():
    score_col = f"{track_type}_score"
    optimized_df[score_col] = score_setups(optimized_df, weights)


# Plot UMAP
def plot_umap(data, color_by="Acceleration"):
    reducer = UMAP(n_neighbors=15, min_dist=0.1)
    X_umap = reducer.fit_transform(data[stat_cols])

    umap_df = pd.DataFrame(X_umap, columns=["UMAP-1", "UMAP-2"])
    umap_df[color_by] = data[color_by].values
    umap_df["Driver"] = data["Driver"]
    umap_df["Vehicle"] = data["Vehicle"]
    umap_df["Tires"] = data["Tires"]
    umap_df["Glider"] = data["Glider"]

    fig = px.scatter(
        umap_df, x="UMAP-1", y="UMAP-2",
        color=color_by,
        color_continuous_scale=px.colors.sequential.Viridis,
        title=f"UMAP Projection Colored by {color_by}",
        hover_data=["Driver", "Vehicle", "Tires", "Glider"]
    )
    fig.show()


# Visualize
plot_umap(optimized_df, color_by="long_straights_score")
plot_umap(optimized_df, color_by="curvy_score")
plot_umap(optimized_df, color_by="offroading_score")
