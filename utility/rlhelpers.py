import random
from collections import namedtuple


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """
        The args here stand for a transition.
        [*arg, arg is a tuple; **kwargs, kwargs is a dict]
        """
        if len(self.memory) < self.capacity:  # There's still some capacity.
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position += 1
        self.position %= self.capacity

    def sample(self, batch_size):
        assert batch_size < self.capacity
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
