import matplotlib.pyplot as plt
import numpy as np

class Bandit:
    # Epsilon greedy bandit
    def __init__(self, true_mean):
        self.true_mean = true_mean
        self.estimated_mean = 0
        self.samples = 0

    def pull(self):
        sample = np.random.randn() + self.true_mean
        return sample

    def update(self, x):
        self.samples += 1
        self.estimated_mean = (1 - 1.0/self.samples)*self.estimated_mean + 1.0/self.samples*x

def run_epsilon_greedy(true_means, epsilon, num_iterations):
    # Initialize the bandits
    bandits = []
    for _, mean in enumerate(true_means):
        bandits.append(Bandit(mean))

    outcomes = np.empty(num_iterations)

    for i in range(num_iterations):
        np.random.seed(i)
        p = np.random.random()

        if p < epsilon:
            # Exploring
            j = np.random.choice(len(bandits))
        else:
            # Exploiting
            j = np.argmax([b.estimated_mean for b in bandits])

        # Updating the corresponding bandit
        x = bandits[j].pull()
        bandits[j].update(x)
        outcomes[i] = x

    cumulative_average = np.cumsum(outcomes) / (np.arange(num_iterations) + 1)
    return cumulative_average

def run_decay_epsilon_greedy(true_means, num_iterations):
    bandits = []
    for _, mean in enumerate(true_means):
        bandits.append(Bandit(mean))

    outcomes = np.empty(num_iterations)

    for i in range(num_iterations):
        np.random.seed(i)
        p = np.random.random()

        # Using epsilon = 1 /t, where t is the current iteration. Adding 1 since we start at 0
        epsilon = 1/(i + 1)

        if p < epsilon:
            # Exploring
            j = np.random.choice(len(bandits))
        else:
            # Exploring
            j = np.argmax([b.estimated_mean for b in bandits])

        # Updating the corresponding bandit
        x = bandits[j].pull()
        bandits[j].update(x)
        outcomes[i] = x

    cumulative_average = np.cumsum(outcomes) / (np.arange(num_iterations) + 1)
    return cumulative_average

class BayesianBandit:
    def __init__(self, true_mean):
        self.true_mean = true_mean

        # Parameters for posterior distribution (assuming that mu ~ N(0,1))
        self.mu_mean = 0
        self.lambda_ = 1

        # Likelihood precision
        self.tau = 1

        # Keeping track of the sum of x's
        self.sum_x = 0

    def pull(self):
        sample = np.random.randn() + self.true_mean
        return sample

    def sample(self):
        sample_posterior = np.random.randn() / np.sqrt(self.lambda_) + self.mu_mean
        return sample_posterior

    def update(self, x):
        self.lambda_ += self.tau
        self.sum_x += x
        self.mu_mean = (self.tau * self.sum_x) / self.lambda_

def run_bayesian(true_means, num_iterations):
    # Initialize the bandits
    bandits = []
    for _, mean in enumerate(true_means):
        bandits.append(BayesianBandit(mean))

    outcomes = np.empty(num_iterations)

    for i in range(num_iterations):
        np.random.seed(i)

        j = np.argmax([b.sample() for b in bandits])
        x = bandits[j].pull()
        bandits[j].update(x)

        outcomes[i] = x

    cumulative_average = np.cumsum(outcomes) / (np.arange(num_iterations) + 1)
    return cumulative_average

np.random.seed(42)
true_means = [1.0, 2.0, 3.0]
num_iterations = 100000

epsilon_1 = 0.01
cumulative_average_ep_1 = run_epsilon_greedy(true_means, epsilon_1, num_iterations)
epsilon_2 = 0.05
cumulative_average_ep_2 = run_epsilon_greedy(true_means, epsilon_2, num_iterations)
epsilon_3 = 0.1
cumulative_average_ep_3 = run_epsilon_greedy(true_means, epsilon_3, num_iterations)

cumulative_average_deg = run_decay_epsilon_greedy(true_means, num_iterations)

cumulative_average_bay = run_bayesian(true_means, num_iterations)

plt.figure()
plt.plot(cumulative_average_ep_1, color = "red", label = "epsilon = 0.01")
plt.plot(cumulative_average_ep_2, color = "blue", label = "epsilon = 0.05")
plt.plot(cumulative_average_ep_3, color = "green", label = "epsilon = 0.1")
plt.plot(cumulative_average_deg, color = "purple", label = "decay")
plt.plot(cumulative_average_bay, color = "orange", label = "bayesian")

plt.xscale('log')
plt.ylabel("Cumulative average of rewards")
plt.xlabel("Iterations (log)")
plt.legend(loc='lower right')
plt.show()
