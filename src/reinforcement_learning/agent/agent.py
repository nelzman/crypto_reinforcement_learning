import keras
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.optimizers import Adam

import time
import numpy as np
import random
from collections import deque

class Agent:
	def __init__(self, state_size, is_eval=False, model_name="", model_path=None):
		self.state_size = state_size # normalized previous days
		self.action_size = 3 # sit, buy, sell
		self.memory = deque(maxlen=1000)
		self.inventory = []
		self.model_name = model_name
		self.model_path = model_path if model_path is not None else "models/"
		self.is_eval = is_eval

		self.gamma = 0.95
		self.epsilon = 1.0
		self.epsilon_min = 0.01
		self.epsilon_decay = 0.995
		self.model = load_model(self.model_path + model_name) if is_eval else self._model()

	def _model(self):
		#model = Sequential()
		#model.add(Dense(units=64, input_dim=self.state_size, activation="relu"))
		#model.add(Dense(units=32, activation="relu"))
		#model.add(Dense(units=8, activation="relu"))
		#model.add(Dense(self.action_size, activation="linear"))
		#model.compile(loss="mse", optimizer=Adam(lr=0.001))

		model = Sequential()
		model.add(Dense(units=64, input_dim=self.state_size, activation="relu"))
		model.add(Dense(units=32, activation="relu"))
		model.add(Dense(units=16, activation="relu"))
		model.add(Dense(units=8, activation="relu"))
		model.add(Dense(self.action_size, activation="linear"))
		model.compile(loss="mse", optimizer=Adam(lr=0.001))

		return model

	def act(self, state):
		if not self.is_eval and random.random() <= self.epsilon:
			return random.randrange(self.action_size)

		options = self.model.predict(state)
		return np.argmax(options[0])

	def expReplay(self, batch_size):
		mini_batch = []
		l = len(self.memory)
		for i in range(l - batch_size + 1, l):
			mini_batch.append(self.memory[i])
		time.sleep(0.1)
		for state, action, reward, next_state, done in mini_batch:
			target = reward
			if not done:
				target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
				#target = reward + self.gamma * np.amax(self.model.predict(next_state)[0]) * (10**(np.max([1,len(str(reward))-2])))
				#target = reward * 1.1				
				#print('Reward: ' + str(reward))
				#print('Target: ' + str(target))
			target_f = self.model.predict(state)
			#print(target_f)
			target_f[0][action] = target
			self.model.fit(state, target_f, epochs=1, verbose=0)

		if self.epsilon > self.epsilon_min:
			self.epsilon *= self.epsilon_decay 
