import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


def update_q_table(Q, s, a, r, sprime, alpha, gamma):
    td_target = r + gamma * np.max(Q[sprime])
    td_error = td_target - Q[s, a]
    Q[s, a] = Q[s, a] + alpha * td_error
    return Q


def epsilon_greedy(Q, s, epsilone):
    if np.random.uniform(0, 1) < epsilone:
        return np.random.randint(len(Q[s]))  
    return np.argmax(Q[s])  


if __name__ == "__main__":

    # TRAIN
    env = gym.make("Taxi-v3", render_mode=None)
    Q = np.zeros([env.observation_space.n, env.action_space.n])

    alpha = 0.7
    gamma = 0.99
    epsilon = 0.2

    n_epochs = 5000
    max_itr_per_epoch = 200
    rewards = []
    
    print("\nTRAINING\n")
    for e in range(n_epochs):
        S, _ = env.reset()
        r = 0
        for _ in range(max_itr_per_epoch):
            A = epsilon_greedy(Q, S, epsilon)
            Sprime, R, done, _, info = env.step(A)
            r += R
            Q = update_q_table(Q, S, A, R, Sprime, alpha, gamma)
            S = Sprime
            if done:
                break
        rewards.append(r)
        if e % 500 == 0:
            print(f"Episode {e}/{n_epochs} -> Reward: {r}")
    print("\nTraining finished.")
    print("Average reward =", np.mean(rewards))

    # TEST
    print("\nTEST\n")
    test_episodes = 5
    epsilon = 0  
    env = gym.make("Taxi-v3", render_mode="human") 
    for i in range(test_episodes):
        S, _ = env.reset()
        r = 0
        for _ in range(max_itr_per_epoch):
            A = np.argmax(Q[S])   
            Sprime, R, done, _, info = env.step(A)
            r += R
            S = Sprime
            if done:
                break
        print(f"Test episode #{i+1} -> Total reward = {r}")
    print("\nTest finished.\n")
    env.close()
