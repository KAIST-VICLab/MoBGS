#!/usr/bin/env python3
#
# File   : safe_ops.py
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

import functools
from typing import Tuple

import jax
import jax.numpy as np


@functools.partial(jax.custom_jvp, nondiff_argnums=(1, 2, 3))
def safe_norm(
    x: np.ndarray,
    axis: int = -1,
    keepdims: bool = False,
    _: float = 1e-9,
) -> np.ndarray:
    """Calculates a np.linalg.norm(d) that's safe for gradients at d=0.

    These gymnastics are to avoid a poorly defined gradient for
    np.linal.norm(0). see https://github.com/google/jax/issues/3058 for details

    Args:
        x (np.ndarray): A np.array.
        axis (int): The axis along which to compute the norm.
        keepdims (bool): if True don't squeeze the axis.
        tol (float): the absolute threshold within which to zero out the
            gradient.

    Returns:
        Equivalent to np.linalg.norm(d)
    """
    return np.linalg.norm(x, axis=axis, keepdims=keepdims)


@safe_norm.defjvp
def _safe_norm_jvp(
    axis: int, keepdims: bool, tol: float, primals: Tuple, tangents: Tuple
) -> Tuple[np.ndarray, np.ndarray]:
    (x,) = primals
    (x_dot,) = tangents
    safe_tol = max(tol, 1e-30)
    y = safe_norm(x, tol=safe_tol, axis=axis, keepdims=True)
    y_safe = np.maximum(y, tol)  # Prevent divide by zero.
    y_dot = np.where(y > safe_tol, x_dot * x / y_safe, np.zeros_like(x))
    y_dot = np.sum(y_dot, axis=axis, keepdims=True)
    # Squeeze the axis if `keepdims` is True.
    if not keepdims:
        y = np.squeeze(y, axis=axis)
        y_dot = np.squeeze(y_dot, axis=axis)
    return y, y_dot


def log1p_safe(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.minimum(x, 3e37))


def exp_safe(x: np.ndarray) -> np.ndarray:
    return np.exp(np.minimum(x, 87.5))


def expm1_safe(x: np.ndarray) -> np.ndarray:
    return np.expm1(np.minimum(x, 87.5))


def safe_sqrt(x: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    safe_x = np.where(x == 0, np.ones_like(x) * eps, x)
    return np.sqrt(safe_x)
