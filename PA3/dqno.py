from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.autograd as autograd
import math, random
USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

class QLearner(nn.Module):
    def __init__(self, env, num_frames, batch_size, gamma, replay_buffer):
        super(QLearner, self).__init__()

        self.batch_size = batch_size
        self.gamma = gamma
        self.num_frames = num_frames
        self.replay_buffer = replay_buffer
        self.env = env
        # input_shape (1, 84, 84), num_actions 6
        self.input_shape = self.env.observation_space.shape
        self.num_actions = self.env.action_space.n

        self.features = nn.Sequential(
            nn.Conv2d(self.input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    def feature_size(self):
            return self.features(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)
    
    def act(self, state, epsilon):
        if random.random() > epsilon:
            state = Variable(torch.FloatTensor(np.float32(state)).unsqueeze(0), requires_grad=True)
            # TODO: Given state, you should write code to get the Q value and chosen action
        
            # output = argmax(self(input))
 



        else:
            action = random.randrange(self.env.action_space.n)
        return action

    def copy_from(self, target):
        self.load_state_dict(target.state_dict())

        
def compute_td_loss(model, target_model, batch_size, gamma, replay_buffer):
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    state = Variable(torch.FloatTensor(np.float32(state)))
    next_state = Variable(torch.FloatTensor(np.float32(next_state)).squeeze(1), requires_grad=True)
    action = Variable(torch.LongTensor(action))
    reward = Variable(torch.FloatTensor(reward))
    done = Variable(torch.FloatTensor(done))
    # implement the loss function here

    action = (B, 1)
    size(done) = (B, 1) 
    terminate = torch.ones_like(done) - done
    r_j = reward
    size(minibatch) is B
    size(next_state) (B, 1, 84, 84) or (B, 84, 84)
    ap = target_model(next_state) = (B, 6)
    # target_model(next_state).max(dim=1)
    gamma is on CPU
    all other things are on GPU.
    Qs = target_model(next_state).detach().cpu().numpy().max(dim=1)
    # Qs = target_model(next_state).max(dim=1)[0].detach().cpu().numpy()
    y_j = r_j + Variable(torch.FloatTensor(gamma * Qs)) * terminate # broadcasting
    Qs = [1.5, 3.6, 2.7]
    terminate = [1.0, 0.0, 1.0]
    Qs * tterminate = [1.5, 0.0, 2.7]
    size(y_j) = (B, 1)

    Q_curr = model(state)
    size(Q_curr) = (B, 6)
    Q = Q_curr[:, action] # broadcasting
    size(Q) = (B, 1)
    loss = (y_j - Q)**2
    loss.mean() * B = loss.sum()
    # Variable(torch.FloatTensor(gamma, requires_grad=False))

    # Q_values = model(state).detach().cpu().numpy()
    
    return loss


class ReplayBuffer(object):
    def __init__(self, capacity):
        # deque: list-like container with fast appends and pops on either end
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        # TODO: Randomly sampling data with specific batch size from the buffer


        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)
