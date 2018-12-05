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
import numpy as np

def npsoftmax(x, axis=None):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def compute_top_k(arr, k):
    assert(k <= np.size(arr))
    return arr.argsort()[-k:][::-1]

def set_pyro_config():
    import Pyro4
    Pyro4.config.SERVERTYPE            = "multiplex"
    Pyro4.config.SERIALIZERS_ACCEPTED  = ["marshal", "json", "serpent", "pickle"]
    Pyro4.config.SERIALIZER            = "pickle"
    Pyro4.config.COMPRESSION           = True
    Pyro4.config.DETAILED_TRACEBACK    = True
