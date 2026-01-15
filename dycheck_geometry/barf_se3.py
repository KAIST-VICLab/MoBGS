#!/usr/bin/env python3
#
# File   : se3.py
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

import jax
from jax import numpy as np

from . import utils


from functools import partial
from math import factorial
from jax import jvp, vmap

# def func_A(x):
#   return np.sin(x)/x

# def func_B(x):
#   return (1-np.cos(x))/x**2

# def func_C(x):
#   return (x-np.sin(x))/x**3

# # a recursive definition shares some work
# def taylor(f, order):
#   def improve_approx(g, k):
#     return lambda x, v: jvp_first(g, (x, v), v)[1] + f(x) / factorial(k)
#   approx = lambda x, v: f(x) / factorial(order)
#   for n in range(order):
#     approx = improve_approx(approx, order - n - 1)
#   return approx

# def jvp_first(f, primals, tangent):
#   x, xs = primals[0], primals[1:]
#   return jvp(lambda x: f(x, *xs), (x,), (tangent,))

def procrustes_analysis(X0, X1):  # [N,3] X0 is target X1 is src
    # translation
    t0 = X0.mean(axis=0, keepdims=True)
    t1 = X1.mean(axis=0, keepdims=True)
    X0c = X0 - t0
    X1c = X1 - t1
    # scale
    s0 = np.sqrt((X0c**2).sum(axis=-1).mean())
    s1 = np.sqrt((X1c**2).sum(axis=-1).mean())
    X0cs = X0c / s0
    X1cs = X1c / s1
    # rotation (use double for SVD, float loses precision)

    U, S, V = np.linalg.svd((X0cs.T @ X1cs), full_matrices=False)
    R = (U @ V.T)

    if np.linalg.det(R) < 0:
        R = R.at[2].set(-R[2])
    sim3 = edict(t0=t0[0], t1=t1[0], s0=s0, s1=s1, R=R)
    return sim3

@jax.jit
def skew(w: np.ndarray) -> np.ndarray:
    """Build a skew matrix ("cross product matrix") for vector w.
    Modern Robotics Eqn 3.30.

    Args:
        w: (..., 3,) A 3-vector

    Returns:
        W: (..., 3, 3) A skew matrix such that W @ v == w x v
    """
    zeros = np.zeros_like(w[..., 0])
    return np.stack(
        [
            np.stack([zeros, -w[..., 2], w[..., 1]], axis=-1),
            np.stack([w[..., 2], zeros, -w[..., 0]], axis=-1),
            np.stack([-w[..., 1], w[..., 0], zeros], axis=-1),
        ],
        axis=-2,
    )

@jax.jit
def taylor_A(x,nth=10):
    # Taylor expansion of sin(x)/x
    ans = np.zeros_like(x)
    denom = 1.
    for i in range(nth+1):
        if i>0: denom *= (2*i)*(2*i+1)
        ans = ans+(-1)**i*x**(2*i)/denom
    return ans

@jax.jit
def taylor_B(x,nth=10):
    # Taylor expansion of (1-cos(x))/x**2
    ans = np.zeros_like(x)
    denom = 1.
    for i in range(nth+1):
        denom *= (2*i+1)*(2*i+2)
        ans = ans+(-1)**i*x**(2*i)/denom
    return ans

@jax.jit
def taylor_C(x,nth=10):
    # Taylor expansion of (x-sin(x))/x**3
    ans = np.zeros_like(x)
    denom = 1.
    for i in range(nth+1):
        denom *= (2*i+2)*(2*i+3)
        ans = ans+(-1)**i*x**(2*i)/denom
    return ans

def se3_to_SE3(w: np.ndarray, u: np.ndarray) -> np.ndarray:
    wx = skew(w)
    theta = np.linalg.norm(w, axis=-1)[...,None,None]
    I = np.identity(3, dtype=w.dtype)

    A = taylor_A(theta)
    B = taylor_B(theta)
    C = taylor_C(theta)

    # check nerfies: R is e^r, V is G, u is v, wx is [r]

    R = I+A*wx+B*wx@wx 
    V = I+B*wx+C*wx@wx

    t = (V@u[...,None])
    return R,t
