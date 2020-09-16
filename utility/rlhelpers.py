import random
from collections import namedtuple
import numpy as np

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


class BalancedReplayMemory(object):
    def __init__(self, capacity, n_option):
        self.capacity = capacity
        self.memory = [[] for _ in range(n_option)]

    def push(self, *args):
        """
        The args here stand for a transition.
        [*arg, arg is a tuple; **kwargs, kwargs is a dict]
        """
        if len(self) == self.capacity:
            longest = 0
            for i, l in enumerate(self.memory):
                if len(self.memory[longest]) < len(l):
                    longest = i
            # print([len(x) for x in self.memory])
            # print(f'arg max proportion action :=> {longest}')
            self.memory[longest].pop(0)
        trans = Transition(*args)
        self.memory[trans.action].append(trans)

    def sample(self, batch_size):
        assert batch_size < self.capacity
        ret = []
        count = np.array([len(x) for x in self.memory])
        index = np.random.choice(count.sum(), size=batch_size, replace=False)
        index = np.sort(index)
        for i in range(1, len(count)):
            count[i] += count[i-1]
        cur = 0
        for ind in index:
            while ind >= count[cur]:
                cur += 1
            ret.append(self.memory[cur][ind - (count[cur-1] if cur > 0 else 0)])
        return ret

    def __len__(self):
        return sum([len(x) for x in self.memory])
