import collections
import random


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
