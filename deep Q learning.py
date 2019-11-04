# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 23:25:26 2019

@author: Daniel Smith
"""

import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from collections import deque
import random
import gym


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        
    
    def _build_model(self):
        # Creating a NN for Deep Q Learning model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model
    
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
        
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])
    
    
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
                
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
            
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
            

EPISODES = 1000
SCORE_GOAL = 500
    
if __name__ == '__main__':
    
    # Initialize gym environment and agent
    env = gym.make('CartPole-v0')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    done = False
    batch_size = 32
    
    # Iterate the game
    for e in range(EPISODES):
        # reset state at the beginning of each game
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        
        # time_t represents each frame of the game
        # We want to achieve at least a score of 500 by holding the pole upright
        for time_t in range(SCORE_GOAL):
            # To render...
            env.render()
            
            # Decide which action to take
            action = agent.act(state)
            
            # Move to the next frame.
            # We will give a score of 1 every frame the pole is upright
            next_state, reward, done, _ = env.step(action)
            
            reward = reward if not done else -10
            
            next_state = np.reshape(next_state, [1, state_size])
            
            # Remember the previous state, action, reward and done
            agent.remember(state, action, reward, next_state, done)
            
            # Make the next_state the new current state for the next frame
            state = next_state
            
            # Done becomes True when the game ends, in this case, when the
            # agent drops the pole from it's prlatform
            if done:
                print('Episodes: {}/{}, score: {}'.format(e, EPISODES, time_t))
                break
            
            if len(agent.memory) > batch_size:
                agent.replay(32)
    
            
            
            
            
            
        
        