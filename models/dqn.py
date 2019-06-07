import collections
import math

import numpy
import numpy.random as random
import torch
import torch.nn.functional as F

from . import segmenttree


GAMMA = 0.99  # discount factor


def pairwise(iterable):
    from itertools import tee
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


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


class PrioritizedReplayMemory:
    def __init__(self, max_len, batch_size, alpha=0.6):
        # self.memory = collections.deque(maxlen=max_len)
        self.memory = [None] * max_len
        self.memory_pos = 0
        self.max_len = max_len

        self.batch_size = batch_size
        self.alpha = alpha

        capacity = 2 ** math.ceil(math.log2(self.max_len))
        self.sum_tree = segmenttree.SumSegmentTree(capacity)
        self.min_tree = segmenttree.MinSegmentTree(capacity)
        self.max_priority = 1.0

    def push(self, transition):
        self.memory[self.memory_pos] = transition

        self.sum_tree[self.memory_pos] = self.max_priority ** self.alpha
        self.min_tree[self.memory_pos] = self.max_priority ** self.alpha

        self.memory_pos = (self.memory_pos + 1) % self.max_len

    def sample(self, beta=0.4):
        prio_segment_len = self.sum_tree() / self.batch_size

        indices = [
            self.sum_tree.get_leaf_idx(
                random.uniform(lower * prio_segment_len, upper * prio_segment_len)
            )
            for lower, upper in pairwise(range(self.batch_size))
        ]
        samples = (memory[idx] for idx in indices)

        max_weight = (len(self.memory) * self.min_tree() / self.sum_tree()) ** (-beta)

        weights = numpy.array(
            [
                (len(self.memory) * self.sum_tree[idx] / self.sum_tree()) ** (-beta)
                for idx in indices
            ]
        )

        weights = weights / max_weight

        return samples, weights, indices

    def __len__(self):
        return len(self.memory)

    def update(self, indices, priorities):
        for idx, prio in zip(indices, priorities):
            self.sum_tree[idx] = prio * self.alpha
            self.min_tree[idx] = prio * self.alpha
        leafs = self.sum_tree.values[self.capacity:]
        self.max_priority = max(leafs + [self.max_priority])


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

        self.fc_adv = torch.nn.Linear(h * w * 32, 512)
        self.adv = torch.nn.Linear(512, outputs)

        self.fc_val = torch.nn.Linear(h * w * 32, 512)
        self.val = torch.nn.Linear(512, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)

        advantage = self.adv(F.relu(self.fc_adv(x)))
        value = self.val(F.relu(self.fc_val(x)))

        return value + advantage - advantage.mean()

    def sample_action(self, state, act_space, epsilon):
        if random.random() < epsilon:
            return torch.tensor([random.randint(0, act_space.n)], device=state.device)

        with torch.no_grad():
            return self.forward(state).max(1).indices


def train(model, target_model, memory, optimizer):
    batch = memory.sample()
    states, actions, rewards, next_states, done = map(torch.cat, zip(*batch))

    q_values = model(states).gather(1, actions.unsqueeze(1))

    q_values_from_next_state = target_model(next_states).max(1).values.detach()
    expected_q_values = rewards + GAMMA * q_values_from_next_state * (1 - done)

    loss = F.smooth_l1_loss(expected_q_values.unsqueeze(1), q_values)

    optimizer.zero_grad()
    loss.backward()
    # for p in model.parameters():  # TODO: test if helps
    #     p.grad.data.clamp_(-1, +1)
    optimizer.step()
