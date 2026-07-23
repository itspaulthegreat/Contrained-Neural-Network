"""
IPOPT iteration callback.

Records the primal iterate at every interior-point iteration so the path taken
by the solver can be inspected or replayed after the solve.
"""

import numpy as np
import casadi as ca


class IterRecorder(ca.Callback):
    """CasADi callback that stores IPOPT's primal iterate x at every iteration."""

    def __init__(self, name, nx, ng):
        ca.Callback.__init__(self)
        self.nx, self.ng = nx, ng
        self.iterates = []
        self.construct(name, {})

    def get_n_in(self):  return ca.nlpsol_n_out()
    def get_n_out(self): return 1
    def get_name_in(self, i):  return ca.nlpsol_out(i)
    def get_name_out(self, i): return 'ret'

    def get_sparsity_in(self, i):
        n = ca.nlpsol_out(i)
        if n == 'f':
            return ca.Sparsity.dense(1)
        if n in ('x', 'lam_x'):
            return ca.Sparsity.dense(self.nx)
        if n in ('g', 'lam_g'):
            return ca.Sparsity.dense(self.ng)
        return ca.Sparsity(0, 0)

    def eval(self, arg):
        self.iterates.append(np.asarray(arg[0]).flatten().copy())
        return [0]
