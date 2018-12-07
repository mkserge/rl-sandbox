import numpy as np
import matplotlib.pyplot as plt

class Bandit:
	def __init__(self, m):
		# The actual mean of the bandit
		self.m = m
		# The observed mean of the bandit
		self.mean = 0
		# Number of times the bandit has been played.
		self.N = 0

	def pull(self):
		# Pull the arm of the bandit. 
		return np.random.randn() + self.m

	def update(self, x):
		# Update the observed mean of the Bandit based on returned value n
		self.N += 1
		self.mean = (1 - 1.0/self.N) * self.mean + 1.0/self.N * x


def plot_bandit_pulls(bandit, N):
	pulls = np.empty(N)
	for i in range(N):
		x = bandit.pull()
		pulls[i] = x
	plt.hist(pulls)
	plt.show()


def run_experiment(m1, m2, m3, eps, N):
	"""Runs an experiment on three bandits, with means m1, m2 and m3.

	Args:
		m1, m2, m3	: mean returned value from the corresponding bandit pull. 
		eps 		: epsilon for the epsilon-greedy strategy. Defined as the 
					  probability of exploration.
		N           : Number of plays.
	Yields:
		The compounded return.
	"""
	# Create the bandits
	reward = np.empty(N)
	bandits = [Bandit(m1), Bandit(m2), Bandit(m3)]
	for turn in range(N):
		# Pick a random number beteween 0 and 1.
		rand = np.random.random()
		if rand < eps:
			# Explore
			# We pick a random bandit.
			bandit = np.random.choice(bandits)
		else:
			# Otherwise, we exploit by taking the best-so-far bandit.
			bandit = max(bandits, key=lambda bandit: bandit.mean)
		x = bandit.pull()
		bandit.update(x)
		reward[turn] = x
	cum_average = np.cumsum(reward) / (np.arange(N) + 1)

	plt.plot(cum_average)
	plt.plot(np.ones(N) * m1)
	plt.plot(np.ones(N) * m2)
	plt.plot(np.ones(N) * m3)
	plt.show()

	# print the final observed means.
	print "Final observed means:"
	print [bandit.mean for bandit in bandits]

	return cum_average


def main():

	c1 = run_experiment(1.0, 2.0, 3.0, 0.01, 100000)
	c2 = run_experiment(1.0, 2.0, 3.0, 0.05, 100000)
	c3 = run_experiment(1.0, 2.0, 3.0, 0.1, 100000)

	plt.plot(c1, label='eps = 0.01')
	plt.plot(c2, label='eps = 0.05')
	plt.plot(c3, label='eps = 0.1')
	plt.legend()
	plt.show()


if __name__ == '__main__':
	main()