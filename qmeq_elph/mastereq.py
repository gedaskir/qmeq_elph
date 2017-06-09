"""Module for solving different master equations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import scipy as sp
import scipy.sparse.linalg
import copy

from qmeq.mytypes import doublenp
from qmeq.mytypes import complexnp

from qmeq import Builder
from qmeq import Transport
from qmeq import StateIndexingDM
from qmeq import QuantumDot
from qmeq import LeadsTunneling
from qmeq.mastereq import FunctionProperties
from .baths import PhononBaths

#-----------------------------------------------------------
# Python modules

from .approach.pauli import elph_generate_paulifct
from .approach.pauli import elph_generate_kern_pauli
from qmeq.approach.pauli import generate_paulifct
from qmeq.approach.pauli import generate_kern_pauli
from qmeq.approach.pauli import generate_current_pauli

from .approach.lindblad import elph_generate_tLbbp
from .approach.lindblad import elph_generate_kern_lindblad
from qmeq.approach.lindblad import generate_tLba
from qmeq.approach.lindblad import generate_kern_lindblad
from qmeq.approach.lindblad import generate_current_lindblad

from .approach.neumann1 import elph_generate_w1fct
from .approach.neumann1 import elph_generate_kern_1vN
from qmeq.approach.neumann1 import generate_phi1fct
from qmeq.approach.neumann1 import generate_kern_1vN
from qmeq.approach.neumann1 import generate_phi1_1vN

# Cython compiled modules

from .approach.c_pauli import c_elph_generate_paulifct
from .approach.c_pauli import c_elph_generate_kern_pauli
from qmeq.approach.c_pauli import c_generate_paulifct
from qmeq.approach.c_pauli import c_generate_kern_pauli
from qmeq.approach.c_pauli import c_generate_current_pauli

from .approach.c_lindblad import c_elph_generate_tLbbp
from .approach.c_lindblad import c_elph_generate_kern_lindblad
from qmeq.approach.c_lindblad import c_generate_tLba
from qmeq.approach.c_lindblad import c_generate_kern_lindblad
from qmeq.approach.c_lindblad import c_generate_current_lindblad

from .approach.c_redfield import c_elph_generate_kern_redfield
from qmeq.approach.c_redfield import c_generate_kern_redfield
from qmeq.approach.c_redfield import c_generate_phi1_redfield

from .approach.c_neumann1 import c_elph_generate_w1fct
from .approach.c_neumann1 import c_elph_generate_kern_1vN
from qmeq.approach.c_neumann1 import c_generate_phi1fct
from qmeq.approach.c_neumann1 import c_generate_kern_1vN
from qmeq.approach.c_neumann1 import c_generate_phi1_1vN
#-----------------------------------------------------------

# Inherit from qmeq.Builder
class Builder_elph(Builder):

    def __init__(self, nsingle, hsingle, coulomb,
                       nleads, tleads, mulst, tlst, dband,
                       nbaths, velph, tlst_ph, dband_ph,
                       indexing='charge', kerntype='Pauli',
                       symq=True, norm_row=0, solmethod='n',
                       itype=0, itype_ph=0, dqawc_limit=10000,
                       mtype_qd=complex, mtype_leads=complex,
                       bath_func=None, eps_elph=1.0e-6):
        '''
        `nbaths', `velph', `tlst_ph', `dband_ph''
        are new parameter for Electron-Phonon coupling
        '''

        if not indexing in {'Lin', 'charge', 'sz', 'ssq'}:
            print("WARNING: Allowed indexing values are: \'Lin\', \'charge\', \'sz\', \'ssq\'. "+
                  "Using default indexing=\'charge\'.")
            indexing = 'charge'

        if not itype in {0,1,2,3}:
            print("WARNING: itype needs to be 0, 1, 2, or 3. Using default itype=0.")
            itype = 0
        if not itype_ph in {0,2}:
            print("WARNING: itype_ph needs to be 0 or 2. Using default itype=0.")
            itype = 0

        if not kerntype in {'Pauli', 'Lindblad', 'Redfield', '1vN', 'pyPauli', 'pyLindblad', 'py1vN'}:
            print("WARNING: Allowed kerntype values are: "+
                  "\'Pauli\', \'Lindblad\', \'Redfield\', \'1vN\', "+
                  "\'pyPauli\', \'pyLindblad\', \'py1vN\', "+
                  "Using default kerntype=\'Pauli\'.")
            kerntype = 'Pauli'

        # Make copies of initialized parameters.
        hsingle = copy.deepcopy(hsingle)
        coulomb = copy.deepcopy(coulomb)
        tleads = copy.deepcopy(tleads)
        mulst = copy.deepcopy(mulst)
        tlst = copy.deepcopy(tlst)
        dband = copy.deepcopy(dband)
        #
        velph = copy.deepcopy(velph)
        tlst_ph = copy.deepcopy(tlst_ph)
        dband_ph = copy.deepcopy(dband_ph)


        funcp = FunctionProperties(kerntype=kerntype, symq=symq, norm_row=norm_row, solmethod=solmethod,
                                   itype=itype, dqawc_limit=dqawc_limit,
                                   mtype_qd=mtype_qd, mtype_leads=mtype_leads, dband=dband)

        si = StateIndexingDM(nsingle, indexing)
        qd = QuantumDot(hsingle, coulomb, si, mtype_qd)
        leads = LeadsTunneling(nleads, tleads, si, mulst, tlst, dband, mtype_leads)
        baths = PhononBaths(nbaths, velph, si, tlst_ph, dband_ph)
        baths.bath_func = bath_func
        tt = Transport_elph(qd, leads, baths, si, funcp)

        self.funcp, self.si, self.qd, self.leads, self.baths, self.tt = funcp, si, qd, leads, baths, tt
        self.funcp.itype_ph = itype_ph
        self.funcp.eps_elph = eps_elph

# Inherit from qmeq.Transport
class Transport_elph(Transport):

    def __init__(self, qd=None, leads=None, baths=None, si=None, funcp=None):
        Transport.__init__(self, qd, leads, si, funcp)
        self.baths = baths

    def set_kern(self):
        kerntype = self.funcp.kerntype
        # Cython modules
        if kerntype == 'Pauli':
            self.paulifct = c_generate_paulifct(self)
            self.kern, self.bvec = c_generate_kern_pauli(self)
            self.paulifct_elph = c_elph_generate_paulifct(self)
            self.kern = c_elph_generate_kern_pauli(self)
        elif kerntype == 'Redfield':
            self.phi1fct, self.phi1fct_energy = c_generate_phi1fct(self)
            self.kern, self.bvec = c_generate_kern_redfield(self)
            self.w1fct = c_elph_generate_w1fct(self)
            self.kern = c_elph_generate_kern_redfield(self)
        elif kerntype == '1vN':
            self.phi1fct, self.phi1fct_energy = c_generate_phi1fct(self)
            self.kern, self.bvec = c_generate_kern_1vN(self)
            self.w1fct = c_elph_generate_w1fct(self)
            self.kern = c_elph_generate_kern_1vN(self)
        elif kerntype == 'Lindblad':
            self.tLba = c_generate_tLba(self)
            self.kern, self.bvec = c_generate_kern_lindblad(self)
            self.tLbbp = c_elph_generate_tLbbp(self)
            self.kern = c_elph_generate_kern_lindblad(self)
        # Python modules
        elif kerntype == 'pyPauli':
            self.paulifct = generate_paulifct(self)
            self.kern, self.bvec = generate_kern_pauli(self)
            self.paulifct_elph = elph_generate_paulifct(self)
            self.kern = elph_generate_kern_pauli(self)
        elif kerntype == 'pyLindblad':
            self.tLba = generate_tLba(self)
            self.kern, self.bvec = generate_kern_lindblad(self)
            self.tLbbp = elph_generate_tLbbp(self)
            self.kern = elph_generate_kern_lindblad(self)
        elif kerntype == 'py1vN':
            self.phi1fct, self.phi1fct_energy = generate_phi1fct(self)
            self.kern, self.bvec = generate_kern_1vN(self)
            self.w1fct = elph_generate_w1fct(self)
            self.kern = elph_generate_kern_1vN(self)

    def solve(self, qdq=True, rotateq=True, masterq=True, currentq=True, *args, **kwargs):
        if qdq:
            self.qd.diagonalise()
            if rotateq:
                self.leads.rotate(self.qd.vecslst)
                self.baths.rotate(self.qd.vecslst)
        #
        if masterq:
            self.set_kern()
            self.solve_kern()
            if currentq:
                self.calc_current()
