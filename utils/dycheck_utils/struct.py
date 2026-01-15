#!/usr/bin/env python3
#
# File   : structures.py
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

from typing import NamedTuple, Optional

from . import types
import numpy as np


class Metadata(NamedTuple):
    time: Optional[np.ndarray] = None
    camera: Optional[np.ndarray] = None
    time_to: Optional[np.ndarray] = None


class Rays(NamedTuple):
    origins: np.ndarray
    directions: np.ndarray
    pixels: np.ndarray
    local_directions: Optional[np.ndarray] = None
    radii: Optional[np.ndarray] = None
    metadata: Optional[Metadata] = None

    near: Optional[np.ndarray] = None
    far: Optional[np.ndarray] = None


class Samples(NamedTuple):
    xs: np.ndarray
    directions: np.ndarray
    cov_diags: Optional[np.ndarray] = None
    metadata: Optional[Metadata] = None

    tvals: Optional[np.ndarray] = None

