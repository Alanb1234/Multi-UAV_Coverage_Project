import matplotlib.pyplot as plt
import numpy as np

# Read the data from the file
episodes = []
rewards = []

with open('logs/rewards.txt', 'r') as f:
    for line in f:
        ep, reward = line.strip().split(',')
        episodes.append(int(ep))
        rewards.append(float(reward))

# Convert to numpy arrays for easier manipulation
episodes = np.array(episodes)
rewards = np.array(rewards)

# Create the plot
plt.figure(figsize=(12, 6))
plt.plot(episodes, rewards, label='Reward')

# Calculate and plot moving average
window_size = 20
moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
plt.plot(episodes[window_size-1:], moving_avg, label=f'{window_size}-Episode Moving Average', color='red')

plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Reward per Episode')
plt.legend()
plt.grid(True)

# Save the plot
plt.savefig('qtran_reward_plot.png')
plt.show()