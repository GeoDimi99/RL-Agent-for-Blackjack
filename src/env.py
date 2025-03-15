# DQN for Black Jack 

# Import necesary libraries
import gymnasium as gym
import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from collections import deque
from collections import defaultdict

class BlackjackAgent:

    # Blackjack environment parameters
    __OBS_SPACE_SIZE=3
    __ACT_SPACE_SIZE=1
    __ACT_NUM=2

    def __init__(
            self,
            env,
            epsilon_max: int = 70,
            epsilon_min: int = 5,
            gamma: float = 0.9,
            learning_rate: float = 0.01,
            dataset_max_size: int = 10000,
            batch_size: int = 1000,
    ):
        self.__env = env
        self.__EPS_MAX = epsilon_max                    	# Initial exploration probability
        self.__epsilon = self.__EPS_MAX						# Current epsilon
        self.__EPS_MIN = epsilon_min                    	# Final expoloration probability 
        self.__GAMMA = gamma                            	# Discount factor
        self.__LR = learning_rate                       	# Learning rate
        self.__EXP_MAX_SIZE = dataset_max_size          	# Experience max size
        self.__BATCH_SIZE = batch_size                  	# Batch size
        self.__experience = deque([], self.__EXP_MAX_SIZE)  # Experience queue
        
        
        self.training_error = []


        # NN architecture for aproximate Q-function
        self.__model = Sequential()
        self.__model.add(Dense(32, input_shape=(self.__OBS_SPACE_SIZE + self.__ACT_SPACE_SIZE,),activation='relu'))
        self.__model.add(Dense(16, activation='relu'))
        self.__model.add(Dense(1,activation='linear'))
        self.__model.compile(optimizer='sgd', loss='mse')

    def get_action(self, obs: tuple[int, int, bool]) -> int:
        rv = random.randint(1,100)
        act = self.__env.action_space.sample()

        if rv >= self.__epsilon:
            candidates = {}
            candidates[0]= self.__model.predict_on_batch(tf.constant([[*obs, 0]]))[0][0]
            candidates[1]= self.__model.predict_on_batch(tf.constant([[*obs, 1]]))[0][0]
            act=max(candidates, key=candidates.get)
        return act

    def update_experience(
            self,
            n,
            obs: tuple [int, int, bool],
            action: int,
            reward: float,
            terminated: bool,
            next_obs: tuple[int, int, bool],
    ):
        candidates_next = {}
        candidates_next[0]= self.__model.predict_on_batch(tf.constant([[*next_obs, 0]]))[0][0]
        candidates_next[1]= self.__model.predict_on_batch(tf.constant([[*next_obs, 1]]))[0][0]
        act_next=max(candidates_next, key=candidates_next.get)

        future_q_value = (not terminated) * candidates_next[act_next]
        current_q_value = self.__model.predict_on_batch(tf.constant([[*obs, action]]))[0][0]


        # Training Rule
        temporal_difference = reward + self.__GAMMA * future_q_value - current_q_value 
        current_q_value = current_q_value + temporal_difference / (n+1)



        if len(self.__experience) >= self.__EXP_MAX_SIZE:
            self.__experience.popleft()

        self.__experience.append([*[*obs, act], current_q_value])
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        self.__epsilon -= self.__epsilon/100
        if self.__epsilon <= self.__EPS_MIN:
            self.__epsilon = self.__EPS_MIN

    def update_nn(
            self,
            n,
    ):
        if len(self.__experience) >= self.__BATCH_SIZE and n % 10 == 0:
            batch = random.sample(self.__experience, self.__BATCH_SIZE)
            trainset = np.array(batch)
            X = trainset[:,:4]
            Y = trainset[:,4]
            
            # train network
            self.__model.fit(tf.constant(X), tf.constant(Y), validation_split=0.2)
            self.decay_epsilon();


# Create blackjack environment
env = gym.make("Blackjack-v1", sab=True)

# Create blackjack agent
agent = BlackjackAgent(env)

n_episodes = 10000
env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=n_episodes)
for episode in tqdm(range(n_episodes)):
	# Initializiate the environment to the first observation
	done = False
	obs, info = env.reset()
	
	while not done:
		# CHOOSE action, EXECUTE action and READ reward and state (obs)
		act = agent.get_action(obs)
		obs_next, reward, terminated, truncated, info = env.step(act)
		
		# update the agent
		agent.update_experience(episode, obs, act, reward, terminated, obs_next)
		
		# update if the environment is done and the current obs
		done = terminated or truncated
		obs = obs_next
	
	# Update neural network 
	agent.update_nn(episode)
	
# Print plot
rolling_length = 500
fig, axs = plt.subplots(ncols=3, figsize=(12, 5))
axs[0].set_title("Episode rewards")
# compute and assign a rolling average of the data to provide a smoother graph
reward_moving_average = (
    np.convolve(
        np.array(env.return_queue).flatten(), np.ones(rolling_length), mode="valid"
    )
    / rolling_length
)
axs[0].plot(range(len(reward_moving_average)), reward_moving_average)
axs[1].set_title("Episode lengths")
length_moving_average = (
    np.convolve(
        np.array(env.length_queue).flatten(), np.ones(rolling_length), mode="same"
    )
    / rolling_length
)
axs[1].plot(range(len(length_moving_average)), length_moving_average)
axs[2].set_title("Training Error")
training_error_moving_average = (
    np.convolve(np.array(agent.training_error), np.ones(rolling_length), mode="same")
    / rolling_length
)
axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)
plt.tight_layout()
plt.show()
	
env.close()
		





