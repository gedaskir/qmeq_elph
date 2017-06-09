"""Module containing python functions, which generate first order Pauli kernel."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import itertools

from qmeq.mytypes import complexnp
from qmeq.mytypes import doublenp
from ..specfunc import Func_pauli_elph


def elph_generate_paulifct(sys):
    (E, Vbbp, si) = (sys.qd.Ea, sys.baths.Vbbp, sys.si)
    func_pauli = Func_pauli_elph(sys.baths.tlst, sys.baths.dlst,
                                 sys.baths.bath_func, sys.funcp.eps_elph)
    #
    paulifct = np.zeros((si.nbaths, si.ndm0, 2), dtype=doublenp)
    for charge in range(si.ncharge):
        # The diagonal elements b=bp are excluded, because they do not contribute
        for b, bp in itertools.combinations(si.statesdm[charge], 2):
            bbp = si.get_ind_dm0(b, bp, charge)
            Ebbp = E[b]-E[bp]
            for l in range(si.nbaths):
                xbbp = (Vbbp[l, b, bp]*Vbbp[l, bp, b]).real
                func_pauli.eval(Ebbp, l)
                paulifct[l, bbp, 1] = xbbp*func_pauli.val
                func_pauli.eval(-Ebbp, l)
                paulifct[l, bbp, 0] = xbbp*func_pauli.val
    return paulifct

#---------------------------------------------------------------------------------------------------------
# Pauli master equation
#---------------------------------------------------------------------------------------------------------
def elph_generate_kern_pauli(sys):
    (paulifct, si, symq, norm_rowp) = (sys.paulifct_elph, sys.si, sys.funcp.symq, sys.funcp.norm_row)
    norm_row = norm_rowp if symq else si.npauli
    last_row = si.npauli-1 if symq else si.npauli
    kern = sys.kern
    if kern is None:
        kern = np.zeros((last_row+1, si.npauli), dtype=doublenp)
    for charge in range(si.ncharge):
        for b in si.statesdm[charge]:
            bb = si.get_ind_dm0(b, b, charge)
            bb_bool = si.get_ind_dm0(b, b, charge, 2)
            if not (symq and bb == norm_row) and bb_bool:
                for a in si.statesdm[charge]:
                    aa = si.get_ind_dm0(a, a, charge)
                    ba = si.get_ind_dm0(b, a, charge)
                    ba_conj = si.get_ind_dm0(b, a, charge, maptype=3)
                    for l in range(si.nbaths):
                        kern[bb, bb] -= paulifct[l, ba, not ba_conj]
                        kern[bb, aa] += paulifct[l, ba, ba_conj]
    return kern
#---------------------------------------------------------------------------------------------------------
