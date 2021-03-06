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
from qmeq import Approach
from qmeq import StateIndexingDM
from qmeq import StateIndexingDMc
from qmeq import QuantumDot
from qmeq import LeadsTunneling
from qmeq import FunctionProperties
from .baths import PhononBaths

#-----------------------------------------------------------
# Python modules

from .approach.pauli import Approach_pyPauli
from .approach.lindblad import Approach_pyLindblad
from .approach.neumann1 import Approach_py1vN
from .approach.redfield import Approach_pyRedfield
from qmeq.approach.neumann2 import Approach_py2vN

# Cython compiled modules

from .approach.c_pauli import Approach_Pauli
from .approach.c_lindblad import Approach_Lindblad
from .approach.c_redfield import Approach_Redfield
from .approach.c_neumann1 import Approach_1vN
from qmeq.approach.c_neumann2 import Approach_2vN
#-----------------------------------------------------------

def check_parameters(indexing, symmetry, itype, kerntype):
    if indexing is 'n':
        if symmetry is 'spin' and kerntype not in {'py2vN', '2vN'}:
            indexing = 'ssq'
        else:
            indexing = 'charge'

    if not indexing in {'Lin', 'charge', 'sz', 'ssq'}:
        print("WARNING: Allowed indexing values are: \'Lin\', \'charge\', \'sz\', \'ssq\'. "+
              "Using default indexing=\'charge\'.")
        indexing = 'charge'

    if not itype in {0,1,2,3}:
        print("WARNING: itype needs to be 0, 1, 2, or 3. Using default itype=0.")
        itype = 0

    if isinstance(kerntype, str):
        if not kerntype in {'Pauli', 'Lindblad', 'Redfield', '1vN', '2vN',
                            'pyPauli', 'pyLindblad', 'pyRedfield', 'py1vN', 'py2vN'}:
            print("WARNING: Allowed kerntype values are: "+
                  "\'Pauli\', \'Lindblad\', \'Redfield\', \'1vN\', \'2vN\', "+
                  "\'pyPauli\', \'pyLindblad\', \'pyRedfield\', \'py1vN\', \'py2vN\'. "+
                  "Using default kerntype=\'Pauli\'.")
            kerntype = 'Pauli'

    if not indexing in {'Lin', 'charge'} and kerntype in {'py2vN', '2vN'}:
        print("WARNING: For 2vN approach indexing needs to be \'Lin\' or \'charge\'. "+
              "Using indexing=\'charge\' as a default.")
        indexing = 'charge'

    return indexing, itype, kerntype

# Inherit from qmeq.Builder
class Builder_elph(Builder):

    def __init__(self, nsingle=0, hsingle={}, coulomb={},
                       nleads=0, tleads={}, mulst={}, tlst={}, dband={},
                       nbaths=0, velph={}, tlst_ph={}, dband_ph={},
                       indexing='n', kpnt=None,
                       kerntype='Pauli', symq=True, norm_row=0, solmethod='n',
                       itype=0, itype_ph=0, dqawc_limit=10000,
                       mfreeq=False, phi0_init=None,
                       mtype_qd=complex, mtype_leads=complex,
                       symmetry='n', herm_hs=True, herm_c=False, m_less_n=True,
                       bath_func=None, eps_elph=1.0e-6):
        '''
        `nbaths', `velph', `tlst_ph', `dband_ph''
        are new parameters for Electron-Phonon coupling
        '''

        indexing, itype, kerntype = check_parameters(indexing, symmetry,
                                                     itype, kerntype)

        if not itype_ph in {0,2}:
            print("WARNING: itype needs to be 0, or 2. Using default itype=0.")
            itype_ph = 0

        if isinstance(kerntype, str):
            self.Approach = globals()['Approach_'+kerntype]
        elif issubclass(kerntype, Approach):
            self.Approach = kerntype
            kerntype = self.Approach.kerntype

        # Make copies of initialized parameters.
        hsingle = copy.deepcopy(hsingle)
        coulomb = copy.deepcopy(coulomb)
        tleads = copy.deepcopy(tleads)
        mulst = copy.deepcopy(mulst)
        tlst = copy.deepcopy(tlst)
        dband = copy.deepcopy(dband)
        phi0_init = copy.deepcopy(phi0_init)
        #
        velph = copy.deepcopy(velph)
        tlst_ph = copy.deepcopy(tlst_ph)
        dband_ph = copy.deepcopy(dband_ph)

        self.funcp = FunctionProperties(symq=symq, norm_row=norm_row, solmethod=solmethod,
                                        itype=itype, dqawc_limit=dqawc_limit,
                                        mfreeq=mfreeq, phi0_init=phi0_init,
                                        mtype_qd=mtype_qd, mtype_leads=mtype_leads,
                                        kpnt=kpnt, dband=dband)
        self.funcp.itype_ph = itype_ph
        self.funcp.eps_elph = eps_elph

        icn = self.Approach.indexing_class_name
        self.si = globals()[icn](nsingle, indexing, symmetry)
        self.qd = QuantumDot(hsingle, coulomb, self.si, herm_hs, herm_c, m_less_n, mtype_qd)
        self.leads = LeadsTunneling(nleads, tleads, self.si, mulst, tlst, dband, mtype_leads)
        self.baths = PhononBaths(nbaths, velph, self.si, tlst_ph, dband_ph)
        self.baths.bath_func = bath_func

        self.appr = self.Approach(self)
        self.create_si_elph()

    def create_si_elph(self):
        si = self.si
        si_elph = StateIndexingDMc(si.nsingle, si.indexing,
                                   si.symmetry, si.nleads)
        si_elph.nbaths = si.nbaths
        self.si_elph = si_elph
        self.appr.si_elph = si_elph

    def remove_states(self, dE):
        Builder.remove_states(self, dE)
        self.si_elph.set_statesdm(si.statesdm)

    # kerntype
    def get_kerntype(self):
        return self.appr.kerntype
    def set_kerntype(self, value):
        if isinstance(value, str):
            if self.appr.kerntype != value:
                self.Approach = globals()['Approach_'+value]
                self.change_si()
                self.appr = self.Approach(self)
                self.create_si_elph()
        else:
            if issubclass(value, Approach):
                self.Approach = value
                self.change_si()
                self.appr = self.Approach(self)
                self.create_si_elph()
    kerntype = property(get_kerntype, set_kerntype)
