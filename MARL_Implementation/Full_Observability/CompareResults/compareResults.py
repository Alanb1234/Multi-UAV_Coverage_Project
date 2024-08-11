import matplotlib.pyplot as plt
import numpy as np

def read_rewards(filename):
    episodes = []
    rewards = []
    with open(filename, 'r') as f:
        for line in f:
            episode, reward = line.strip().split(',')
            episodes.append(int(episode))
            rewards.append(float(reward))
    return episodes, rewards

def running_average(data, window_size):
    return np.convolve(data, np.ones(window_size), 'valid') / window_size

# Read the reward data
idqn_episodes, idqn_rewards = read_rewards('rewards_idqn.txt')
vdn_episodes, vdn_rewards = read_rewards('rewards_vdn.txt')
qmix_episodes, qmix_rewards = read_rewards('rewards_qmix.txt')
qtran_episodes, qtran_rewards = read_rewards('rewards_qtran.txt')

# Calculate running averages
window_size = 20
idqn_avg = running_average(idqn_rewards, window_size)
vdn_avg = running_average(vdn_rewards, window_size)
qmix_avg = running_average(qmix_rewards, window_size)
qtran_avg = running_average(qtran_rewards, window_size)

# Create the plot
plt.figure(figsize=(12, 6))

# Plot raw data
plt.plot(idqn_episodes, idqn_rewards, alpha=0.3, color='blue', label='IDQN')
plt.plot(vdn_episodes, vdn_rewards, alpha=0.3, color='red', label='VDN')
plt.plot(qmix_episodes, qmix_rewards, alpha=0.3, color='green', label='QMIX')
plt.plot(qtran_episodes, qtran_rewards, alpha=0.3, color='black', label='QTRAN')

# Plot running averages
plt.plot(idqn_episodes[window_size-1:], idqn_avg, color='blue', linewidth=2, label='IDQN 20-ep avg')
plt.plot(vdn_episodes[window_size-1:], vdn_avg, color='red', linewidth=2, label='VDN 20-ep avg')
plt.plot(qmix_episodes[window_size-1:], qmix_avg, color='green', linewidth=2, label='QMIX 20-ep avg')
plt.plot(qtran_episodes[window_size-1:], qtran_avg, color='black', linewidth=2, label='QTRAN 20-ep avg')

plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Full_Observability_MARL_Algorithms_Comparison')
plt.legend()
plt.grid(True, alpha=0.3)

# Save the plot
plt.savefig('MARL_Algorithms_Comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("Plot saved as 'MARL_Algorithms_Comparison.png'")