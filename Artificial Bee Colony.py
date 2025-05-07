import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import imageio.v2 as imageio
import os

# Parameters
num_food_sources = 10
num_employed_bees = num_food_sources
num_onlooker_bees = 5
num_scout_bees = 1
dim = 2
limit = 5
iterations = 50
interpolation_steps = 5

# Fitness function: sphere (minimise, so we negate it)
fitness = lambda x: -np.sum(x ** 2)

# Initialisation
food_sources = np.random.uniform(-50, 50, (num_food_sources, dim))
scores = np.array([fitness(fs) for fs in food_sources])
trial_counters = np.zeros(num_food_sources)

# Fitness tracking
best_fitness_over_time = []

# Visualisation setup
fig, (ax, ax_fitness) = plt.subplots(1, 2, figsize=(14, 6))

# Main bee animation plot
ax.set_xlim(-60, 60)
ax.set_ylim(-60, 60)
ax.set_title("Artificial Bee Colony Algorithm", fontsize=16)
ax.grid(True)

scat_food = ax.scatter([], [], s=300, c='gold', edgecolors='black', label='Food Source', zorder=2)
scat_employed = ax.scatter([], [], s=80, c='blue', marker='^', label='Employed Bee', zorder=3)
scat_onlooker = ax.scatter([], [], s=80, c='green', marker='o', label='Onlooker Bee', zorder=3)
scat_scout = ax.scatter([], [], s=100, c='red', marker='x', label='Scout Bee', zorder=3)

ax.legend(loc='upper right')
texts = []

# Fitness curve plot
ax_fitness.set_title("Best Fitness Over Time", fontsize=14)
ax_fitness.set_xlabel("Iteration")
ax_fitness.set_ylabel("Best Fitness")
ax_fitness.grid(True)
fitness_line, = ax_fitness.plot([], [], lw=2, c='purple')

# Store positions between frames for interpolation
prev_employed_pos = np.copy(food_sources[:num_employed_bees])
prev_onlooker_pos = np.random.uniform(-50, 50, (num_onlooker_bees, dim))
prev_scout_pos = np.random.uniform(-50, 50, (num_scout_bees, dim))

interpolated_employed = []
interpolated_onlooker = []
interpolated_scout = []

def interpolate_positions(old, new, steps):
    return [old + (new - old) * (i / steps) for i in range(1, steps + 1)]

def update(frame):
    global food_sources, scores, trial_counters
    global prev_employed_pos, prev_onlooker_pos, prev_scout_pos
    global interpolated_employed, interpolated_onlooker, interpolated_scout

    subframe = frame % interpolation_steps
    iteration = frame // interpolation_steps

    if subframe == 0:
        employed_bees_pos = []
        onlooker_bees_pos = []
        scout_bees_pos = []

        # --- Employed Bee Phase ---
        for i in range(num_employed_bees):
            k = np.random.choice([x for x in range(num_food_sources) if x != i])
            phi = (np.random.rand(dim) - 0.5) * 2
            new_sol = food_sources[i] + phi * (food_sources[i] - food_sources[k])
            new_score = fitness(new_sol)
            if new_score > scores[i]:
                food_sources[i] = new_sol
                scores[i] = new_score
                trial_counters[i] = 0
            else:
                trial_counters[i] += 1
            employed_bees_pos.append(food_sources[i] + np.random.uniform(-2, 2, dim))

        # --- Onlooker Bee Phase ---
        prob = (scores - np.min(scores)) / (np.max(scores) - np.min(scores) + 1e-10)
        prob /= np.sum(prob)
        for _ in range(num_onlooker_bees):
            i = np.random.choice(range(num_food_sources), p=prob)
            k = np.random.choice([x for x in range(num_food_sources) if x != i])
            phi = (np.random.rand(dim) - 0.5) * 2
            new_sol = food_sources[i] + phi * (food_sources[i] - food_sources[k])
            new_score = fitness(new_sol)
            if new_score > scores[i]:
                food_sources[i] = new_sol
                scores[i] = new_score
                trial_counters[i] = 0
            else:
                trial_counters[i] += 1
            onlooker_bees_pos.append(food_sources[i] + np.random.uniform(-3, 3, dim))

        # --- Scout Bee Phase ---
        scout_bees_pos.clear()
        for i in range(num_scout_bees):
            scout_origin = prev_scout_pos[i] if i < len(prev_scout_pos) else np.random.uniform(-50, 50, dim)
            scout_target = np.random.uniform(-50, 50, dim)
            scout_bees_pos.append(scout_target)
        prev_scout_pos = np.array(scout_bees_pos)

        if not scout_bees_pos:
            scout_bees_pos = prev_scout_pos.tolist()
        else:
            prev_scout_pos = np.array(scout_bees_pos)

        # Record best fitness
        best_fitness_over_time.append(np.max(scores))

        # Generate interpolated paths
        interpolated_employed = interpolate_positions(prev_employed_pos, np.array(employed_bees_pos), interpolation_steps)
        interpolated_onlooker = interpolate_positions(prev_onlooker_pos, np.array(onlooker_bees_pos), interpolation_steps)
        interpolated_scout = interpolate_positions(prev_scout_pos, np.array(scout_bees_pos), interpolation_steps)

        prev_employed_pos = np.array(employed_bees_pos)
        prev_onlooker_pos = np.array(onlooker_bees_pos)

    # --- Drawing ---
    scat_food.set_offsets(food_sources)
    scat_employed.set_offsets(interpolated_employed[subframe])
    scat_onlooker.set_offsets(interpolated_onlooker[subframe])
    scat_scout.set_offsets(interpolated_scout[subframe])

    # Update food source labels
    global texts
    for txt in texts:
        txt.remove()
    texts = []
    for i, (x, y) in enumerate(food_sources):
        texts.append(ax.text(x + 1, y + 1, f'F{i}', fontsize=9, color='black'))

    ax.set_xlabel(f"Iteration {iteration + 1}", fontsize=12)

    # Update fitness plot
    fitness_line.set_data(range(len(best_fitness_over_time)), best_fitness_over_time)
    ax_fitness.set_xlim(0, iterations)
    ax_fitness.set_ylim(min(best_fitness_over_time) * 1.1, 0)

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=iterations * interpolation_steps, interval=100, repeat=False)

# Save the animation as a GIF
ani.save('abc_animation.gif', writer='imagemagick', fps=20)

print("GIF saved as 'abc_animation.gif'")
