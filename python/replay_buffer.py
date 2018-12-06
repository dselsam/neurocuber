# Copyright 2018 Daniel Selsam. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
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
        if (not self.eviction_started) and self.next_index < self.cfg['replay_buffer_min_size']:
            # to prevent overfitting the first run that happens to get added
            return []

        n_to_consider = len(self.storage) if self.eviction_started else self.next_index
        if n_to_consider >= n_samples:
            return random.sample(self.storage[:n_to_consider], k=n_samples)
        else:
            return []
