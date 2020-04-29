from dqn import Agent
import numpy as np
import gym
import matplotlib.pyplot as plt


if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    n_games = 300
    show = False
    agent = Agent(gamma=0.99, epsilon=1.0, alpha=0.0005, input_dims=8,
                n_actions=4, batch_size=64)

    scores = []
    eps_history = []

    for i in range(1, n_games+1):
        done = False
        score = 0
        obseervation = env.reset()
        while not done:
            if show:
                env.render()
            action = agent.choose_action(obseervation)
            obseervation_, reward, done, info = env.step(action)
            score += reward
            agent.remember(obseervation, action, reward, obseervation_, done)
            obseervation = obseervation_
            agent.learn()

        eps_history.append(agent.epsilon)
        scores.append(score)

        avg_score = np.mean(scores[max(0, i-100):i+1])
        print('epsiode', i, 'score ', score, 'avg score', avg_score)
        if i % 10 == 0 and i > 0:
            agent.save_model()

    plt.plot(scores)
    plt.plot(eps_history)
    plt.legend(['score', 'epsilon'], loc='upper left')
    plt.show()