import random
import time

class ReplayBuffer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.storage           = [None for _ in range(cfg['replay_buffer_size'])]
        self.next_index        = 0
        self.eviction_started  = False

    def _increment_index(self):
        self.next_index = (self.next_index + 1) % self.cfg['replay_buffer_size']
        if self.next_index == 0 and not self.eviction_started:
            self.eviction_started = True
            print("[REPLAY_BUFFER] Starting eviction...")

    def add_datapoints(self, datapoints):
        if not datapoints: return

        for datapoint in datapoints:
            self.storage[self.next_index] = datapoint
            self._increment_index()

    def sample_datapoints(self, n_samples):
        n_to_consider = len(self.storage) if self.eviction_started else self.next_index
        if n_to_consider >= n_samples:
            return random.sample(self.storage[:n_to_consider], k=n_samples)
        else:
            return []
