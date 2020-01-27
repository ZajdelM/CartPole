import random
import gym
import numpy
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

STEPS = 500

GENERATIONS = 100
EXPERIENCE_LEARNING_BATCH_SIZE = 32
MINIMAL_PROBES = 100

class Agent:
    def __init__(self, neuralNet):
        self.replayBuffer = deque(maxlen=2000)
        self.discountFactor = 0.95
        self.explorationRate = 1.0
        self.explorationRateMin = 0.01
        self.explorationRateMultiplier = 0.999
        self.neuralNet = neuralNet

    def add(self, state, action, reward, nextState, done):
        self.replayBuffer.append((state, action, reward, nextState, done))

    def lowerExplorationRate(self):
        if len(self.replayBuffer) > MINIMAL_PROBES:
            if self.explorationRate > self.explorationRateMin:
                self.explorationRate = self.explorationRate * self.explorationRateMultiplier

    def getAction(self, state):
        if numpy.random.rand() <= self.explorationRate:
            return random.randrange(2)
        actionValues = self.neuralNet.predict(state)
        return numpy.argmax(actionValues[0])

    def experienceReplay(self, batch_size):
        batch = random.sample(self.replayBuffer, batch_size)
        for state, action, reward, nextState, done in batch:
            target = reward
            if not done:
                target = (reward + self.discountFactor *
                          numpy.amax(self.neuralNet.predict(nextState)[0]))
            targetFit = self.neuralNet.predict(state)
            targetFit[0][action] = target
            self.neuralNet.fit(state, targetFit, epochs=1, verbose=0)


def buildNeuralNet():
    model = Sequential()
    model.add(Dense(64, input_dim=4, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(2, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(lr=0.0005))
    return model


env = gym.make('CartPole-v1')
neuralNet = buildNeuralNet()
agent = Agent(neuralNet)
done = False
lastGenWon = False
result = 0
for generation in range(GENERATIONS):
    state = env.reset()
    state = numpy.reshape(state, (1, 4))
    lastGenWon = True
    for i in range(STEPS):
        env.render()
        action = agent.getAction(state)
        nextState, reward, done, _ = env.step(action)
        if not done or i == STEPS - 1:
            reward = reward
        else:
            reward = -100
        nextState = numpy.reshape(nextState, (1, 4))
        agent.add(state, action, reward, nextState, done)
        agent.lowerExplorationRate()
        state = nextState
        if done:
            print("Generation: {}\t Score: {}".format(generation + 1, i))
            lastGenWon = False
            result += i+1
            break
        if MINIMAL_PROBES < len(agent.replayBuffer):
            agent.experienceReplay(EXPERIENCE_LEARNING_BATCH_SIZE)
    if lastGenWon:
        print("Generation: {} won!".format(generation + 1))
        result += 500

print(result/GENERATIONS)