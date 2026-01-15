#!/usr/bin/env python3
#
# File   : utils.py
# Author : Hang Gao
# Email  : hangg.sv7@gmail.com
#
# Copyright 2022 Adobe. All rights reserved.
#
# This file is licensed to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR REPRESENTATIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

import numpy as np

from utils.dycheck_utils import types


def matmul(a: types.Array, b: types.Array) -> types.Array:
    if isinstance(a, np.ndarray):
        assert isinstance(b, np.ndarray)
    else:
        assert isinstance(a, np.ndarray)
        assert isinstance(b, np.ndarray)

    if isinstance(a, np.ndarray):
        return a @ b
    else:
        # NOTE: The original implementation uses highest precision for TPU
        # computation. Since we are using GPUs only, comment it out.
        #  return np.matmul(a, b, precision=jax.lax.Precision.HIGHEST)
        return np.matmul(a, b)


def matv(a: types.Array, b: types.Array) -> types.Array:
    return matmul(a, b[..., None])[..., 0]

