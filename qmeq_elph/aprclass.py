from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.sparse.linalg
from scipy import optimize

from qmeq import Approach

from qmeq.mytypes import doublenp
from qmeq.mytypes import complexnp

# Inherit from qmeq.Approach
class Approach_elph(Approach):

    @staticmethod
    def generate_fct_elph(sys): pass
    @staticmethod
    def generate_kern_elph(sys): pass

    def __init__(self, sys):
        Approach.__init__(self, sys)
        self.baths = sys.baths

    def solve(self, qdq=True, rotateq=True, masterq=True, currentq=True, *args, **kwargs):
        if qdq:
            self.qd.diagonalise()
            if rotateq:
                self.leads.rotate(self.qd.vecslst)
                self.baths.rotate(self.qd.vecslst)
        #
        if masterq:
            self.generate_fct(self)
            self.generate_fct_elph(self)
            if self.funcp.mfreeq:
                self.solve_matrix_free()
            else:
                self.generate_kern(self)
                self.generate_kern_elph(self)
                self.solve_kern()
            if currentq:
                self.generate_current(self)
