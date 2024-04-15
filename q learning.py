import sys

import gymnasium as gym
import imageio
from IPython.display import Image, display
import random


env = gym.make('Taxi-v3', render_mode = "rgb_array")

observation, info = env.reset()

print("Action space:",env.action_space)
print("Observation space:",env.observation_space)
# creating Q table
Q_taxi = {}
possible_states_taxi = range(env.observation_space.n)
possible_actions_taxi = range(env.action_space.n)

for s in possible_states_taxi:
    Q_taxi[s] = {}
    for a in possible_actions_taxi:
        Q_taxi[s][a] = 0

count = 0
# preparing for Q learning equation
state = 0
alpha = 0.1
gamma = 0.6

num_iterations = 1000000
for i in range(num_iterations):
# exploitation implementation
    if random.random() < i/num_iterations:
      best_estimated_reward = float("-inf")
      action = None # initializing action

      for a in possible_actions_taxi:
          if Q_taxi[state][a] > best_estimated_reward:
              best_estimated_reward = Q_taxi[state][a]
              action = a
    else:
      action = random.choice(range(env.action_space.n))

    observation, reward, terminated, truncated, info = env.step(action)
    next_state = observation

# finding the best q val
    next_state_Q_val_list = Q_taxi[next_state].values()
    next_state_bestq = max(next_state_Q_val_list)

# q learning update rule
    Q_taxi[state][action] = Q_taxi[state][action] + alpha*(reward + gamma*(next_state_bestq) - Q_taxi[state][action])

    state = observation # saving the current state for the next loop iteration

    if terminated or truncated:
        count += 1
        observation, info = env.reset()

env.close()
print(count)

frames = []

# preparing for Q learning equation
state = 0
alpha = 0.1
gamma = 0.6

num_iterations = 100
for i in range(num_iterations):
    frames.append( env.render() ) # render the next frame, append to frames list

# exploitation implementation
    if random.random() < i/num_iterations:
      best_estimated_reward = float("-inf")
      action = None # initializing action

      for a in possible_actions_taxi:
          if Q_taxi[state][a] > best_estimated_reward:
              best_estimated_reward = Q_taxi[state][a]
              action = a
    else:
      action = random.choice(range(env.action_space.n))

    observation, reward, terminated, truncated, info = env.step(action)
    next_state = observation

# pre-trained agent and gif display
    next_state_Q_val_list = Q_taxi[next_state].values()
    next_state_bestq = max(next_state_Q_val_list)


    Q_taxi[state][action] = Q_taxi[state][action] + alpha*(reward + gamma*(next_state_bestq) - Q_taxi[state][action])

    state = observation # saving the current state for the next loop iteration

    if terminated or truncated:
        observation, info = env.reset()


env.close()
# save the animation as animation.gif and then display it in the notebook
imageio.mimsave('taxi.gif', frames, fps=30)  # fps: frames per second
display(Image(filename='taxi.gif'))