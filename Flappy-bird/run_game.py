import cv2
import numpy as np
import pickle
import game.wrapped_flappy_bird as game

from PIL import Image

from BrainDQN import BrainDQN as Brain


def PIL2array(img, shape):
    return np.array(img.getdata(), np.uint8).reshape(shape)


def array2PIL(observation):
    # load image using PIL and process it
    img = Image.fromarray(observation).resize((80, 80)).convert('1')
    return img


# def preprocess(observation, shape=(80, 80, 1)):
#     return PIL2array(array2PIL(observation), shape)

def preprocess(observation, shape=(80, 80, 1)):
    observation = cv2.cvtColor(cv2.resize(observation, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, observation = cv2.threshold(observation,1,255,cv2.THRESH_BINARY)
    return np.reshape(observation, shape)


def playFlappyBird():
    flappyBird = game.GameState()

    action0 = np.array([1, 0])  # do nothing
    observation0, reward0, terminal = flappyBird.frame_step(action0)
    observation0 = preprocess(observation0, shape=(80, 80))

    try:
        with open('replayMemory.pkl', 'rb') as f:
            brain = pickle.load(f)
            print('load saved brain')
    except FileNotFoundError:
        print('cannot find saved brain, create a new brain')
        brain = Brain()
        brain.setInitState(observation0)

    while True:
        action = brain.getAction()
        nextObservation, reward, terminal = flappyBird.frame_step(action)
        nextObservation = preprocess(nextObservation)
        brain.setPerception(nextObservation, action, reward, terminal)


playFlappyBird()
