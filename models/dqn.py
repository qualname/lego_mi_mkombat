import collections
import math
import random

import torch
import torch.nn.functional as F

GAMMA = 0.99  # discount factor


class ReplayMemory:
    def __init__(self, max_len, batch_size):
        self.memory = collections.deque(maxlen=max_len)
        self.batch_size = batch_size

    def push(self, transition):
        self.memory.append(transition)

    def sample(self):
        return random.sample(self.memory, self.batch_size)

    def __len__(self):
        return len(self.memory)


class QNN(torch.nn.Module):
    def __init__(self, input_height, input_width, input_frames, outputs):
        super().__init__()

        inp_channels = input_frames * 3  # RGB images

        self.conv1 = torch.nn.Conv2d(inp_channels, 16, kernel_size=7, stride=2)
        self.bn1 = torch.nn.BatchNorm2d(16)
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=7, stride=2)
        self.bn2 = torch.nn.BatchNorm2d(32)
        self.conv3 = torch.nn.Conv2d(32, 32, kernel_size=7, stride=2)
        self.bn3 = torch.nn.BatchNorm2d(32)

        def conv_output_size(size, kernel_size=7, stride=2):
            return math.ceil((size - kernel_size + 1) / stride)

        h = conv_output_size(conv_output_size(conv_output_size(input_height)))
        w = conv_output_size(conv_output_size(conv_output_size(input_width)))

        self.fc4 = torch.nn.Linear(h * w * 32, 512)
        self.head = torch.nn.Linear(512, outputs)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        return self.head(x)

    def sample_action(self, state, act_space, epsilon):
        if random.random() < epsilon:
            return torch.tensor([random.randrange(act_space.n)], device=state.device)

        with torch.no_grad():
            return self.forward(state).max(1).indices


def train(model, target_model, memory, optimizer):
    batch = memory.sample()
    states, actions, rewards, next_states, done = map(torch.cat, zip(*batch))

    q_values = model(states).gather(1, actions.unsqueeze(1))

    q_values_from_next_state = target_model(next_states).gather(
        1, torch.max(model(next_states), 1).indices.unsqueeze(1)
    )
    q_values_from_next_state = q_values_from_next_state.squeeze().detach()
    expected_q_values = rewards + GAMMA * q_values_from_next_state * (1 - done)

    loss = F.smooth_l1_loss(expected_q_values.unsqueeze(1), q_values)

    optimizer.zero_grad()
    loss.backward()
    # for p in model.parameters():  # TODO: test if helps
    #     p.grad.data.clamp_(-1, +1)
    optimizer.step()
