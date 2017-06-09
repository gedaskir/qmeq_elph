"""Module for defining tunneling amplitudes from quantum dot to leads."""

from __future__ import division
import numpy as np
import itertools

from qmeq.indexing import szrange
from qmeq.indexing import ssqrange
from qmeq.indexing import sz_to_ind
from qmeq.indexing import ssq_to_ind
from qmeq.leadstun import construct_full_pmtr
from qmeq.leadstun import make_array
from qmeq.leadstun import make_array_dlst

def elph_construct_Vbbp(velph, stateind, mtype=complex, Vbbp_=None):
    """
    Constructs many-body electron-phonon coupling matrix Vbbp
    from single particle electron-phonon matrix elements Vij.

    Parameters
    ----------
    velph : dict
        Dictionary containing electron-phonon coupling written in single particle basis.
        velph[(bath, i, j)] = Vij, where i, j are the state labels.
    stateind : StateIndexing
        StateIndexing or StateIndexingDM object.
    mtype : type
        Defines type of Vbbp matrix. For example, float, complex, etc.
    Vbbp_ : None or array
        nbaths by nmany by nmany numpy array containing old values of Vbbp.
        The values in velph are added to Vbbp\_.

    Returns
    -------
    Vbbp : array
        nbaths by nmany by nmany numpy array containing many-body electron-phonon coupling matrix.
        The returned Vbbp corresponds to Fock basis.
    """
    if Vbbp_ is None:
        Vbbp = np.zeros((stateind.nbaths, stateind.nmany, stateind.nmany), dtype=mtype)
    else:
        Vbbp = Vbbp_
    # Iterate over many-body states
    for j1 in range(stateind.nmany):
        state = stateind.get_state(j1)
        # Iterate over single particle states
        for j0 in velph:
            (j4, j3, j2), vamp = j0, velph[j0]
            if j2 == j3:
                if state[j2] == 1:
                    Vbbp[j4, j1, j1] += vamp
            # Remove particle from j2 single particle state, add particle in j3 single particle state
            elif state[j2] == 1 and state[j3] == 0:
                # Calculate fermion sign for added/removed electrons in a given state
                # Note that if j3 is larger than j2 additional sign appears for flipping j3 with j2
                fsign = np.power(-1, sum(state[0:j2])+sum(state[0:j3])) * (+1 if j2 > j3 else -1)
                statep = list(state)
                statep[j2] = 0
                statep[j3] = 1
                ind = stateind.get_ind(statep)
                Vbbp[j4, ind, j1] += vamp*fsign
                Vbbp[j4, j1, ind] += vamp.conjugate()*fsign
    return Vbbp

def elph_rotate_Vbbp(Vbbp0, vecslst, stateind, indexing='n', mtype=complex):
    """
    Rotates electron-phonon coupling matrix Vbbp0 in Fock basis to Vbbp,
    which is in eigenstate basis of the quantum dot.

    Parameters
    ----------
    Vbbp0 : array
        nbaths by nmany by nmany numpy array, giving tunneling amplitudes in Fock basis.
    stateind : StateIndexing
        StateIndexing or StateIndexingDM object.
    indexing : string
        Specifies what kind of rotation procedure to use. Default is stateind.indexing.
    mtype : type
        Defines type of Vbbp matrix. For example, float, complex, etc.

    Returns
    -------
    Vbbp : array
        nbaths by nmany by nmany numpy array containing many-body electron-phonon coupling matrix.
        The returned Vbbp corresponds to the quantum dot eigenbasis.
    """
    if indexing == 'n':
        indexingp = stateind.indexing
    else:
        indexingp = indexing
    Vbbp = np.zeros((stateind.nbaths, stateind.nmany, stateind.nmany), dtype=mtype)
    if indexingp == 'Lin':
        pmtr = construct_full_pmtr(vecslst, stateind, mtype)
        for l in range(stateind.nbaths):
            # Calculate many-body tunneling matrix Vbbp=P^(-1).Xbbp0.P
            # in eigenbasis of Hamiltonian from tunneling matrix Vbbp0 in Fock basis.
            # pmtr.conj().T denotes the conjugate transpose of pmtr.
            Vbbp[l] = np.dot(pmtr.conj().T, np.dot(Vbbp0[l], pmtr))
    elif indexingp == 'sz':
        for l, charge in itertools.product(range(stateind.nbaths), range(stateind.ncharge)):
            szrng = szrange(charge, stateind.nsingle)
            for sz in szrng:
                szind = sz_to_ind(sz, charge, stateind.nsingle)
                i1 = stateind.szlst[charge][szind][0]
                i2 = stateind.szlst[charge][szind][-1] + 1
                Vbbp[l, i1:i2][:, i1:i2] = np.dot(vecslst[charge][szind].conj().T, np.dot(Vbbp0[l, i1:i2][:, i1:i2], vecslst[charge][szind]))
    elif indexingp == 'ssq':
        for l, charge in itertools.product(range(stateind.nbaths), range(stateind.ncharge)):
            szrng = szrange(charge, stateind.nsingle)
            for sz in szrng:
                szind = sz_to_ind(sz, charge, stateind.nsingle)
                i1 = stateind.szlst[charge][szind][0]
                i2 = stateind.szlst[charge][szind][-1] + 1
                vecslst1 = np.concatenate(vecslst[charge][szind], axis=1)
                Vbbp[l, i1:i2][:, i1:i2] = np.dot(vecslst1.conj().T, np.dot(Vbbp0[l, i1:i2][:, i1:i2], vecslst1))
    elif indexingp == 'charge':
        for l, charge in itertools.product(range(stateind.nbaths), range(stateind.ncharge)):
            i1 = stateind.chargelst[charge][0]
            i2 = stateind.chargelst[charge][-1] + 1
            Vbbp[l, i1:i2][:, i1:i2] = np.dot(vecslst[charge].conj().T, np.dot(Vbbp0[l, i1:i2][:, i1:i2], vecslst[charge]))
    return Vbbp
#---------------------------------------------------------------------------------------------------

def make_velph_dict(velph):
    """
    Makes single-particle electron-phonon coupling dictionary.

    Parameters
    ----------
    velph : list, dict, or array
        Contains single-particle electron-phonon couplings.

    Returns
    -------
    velph_dict : dictionary
        Dictionary containing electron-phonon couplings.
        velph[(bath, state1, state2)] gives the coupling.
    """
    if isinstance(velph, list):
        velph_dict = {}
        for j0 in velph:
            j1, j2, j3, vamp = j0
            velph_dict.update({(j1, j2, j3):vamp})
        return velph_dict
    elif isinstance(velph, np.ndarray):
        nbaths, nsingle1, nsingle2 = velph.shape
        velph_dict = {}
        for j1, j2, j3 in itertools.product(range(nbaths), range(nsingle1), nsingle2):
            if velph[j1, j2, j3] != 0:
                velph_dict.update({(j1, j2, j3):velph[j1, j2, j3]})
        return velph_dict
    elif isinstance(velph, dict):
        return velph
#---------------------------------------------------------------------------------------------------

class PhononBaths(object):
    """
    Class for defining electron-phonon couplings from baths to the quantum dot.

    Attributes
    ----------
    nbaths : int
        Number of the phonon baths.
    velph : list, dict, or array
        list, dictionary or numpy array defining single-particle electron-phonon couplings.
        numpy array has to be nbaths by nsingle.
    stateind : StateIndexing
        StateIndexing or StateIndexingDM object.
    tlst : list
        List containing temperatures of the phonon baths.
    dlst : list
        List containing bandwidths of the phonon baths.
    mtype : type
        Defines type of Vbbp0 and Vbbp matrices. For example, float, complex, etc.
    Vbbp0 : array
        nmany by nmany array, which contains many-body electron-phonon coupling matrix in Fock basis.
    Vbbp : list
        nmany by nmany array, which contains many-body electron-phonon coupling matrix,
        which is used in calculations.
    """

    def __init__(self, nbaths, velph, stateind, tlst, dlst, mtype=complex):
        """Initialization of the LeadsTunneling class."""
        self.velph = make_velph_dict(velph)
        self.stateind = stateind
        self.stateind.nbaths = nbaths
        self.tlst = make_array(tlst, nbaths)
        self.dlst = make_array_dlst(dlst, nbaths)
        self.mtype = mtype
        self.Vbbp0 = elph_construct_Vbbp(self.velph, stateind, mtype)
        self.Vbbp = self.Vbbp0
        self.bath_func = None

    def add(self, velph=None, tlst=None, dlst=None, updateq=True, lstq=True):
        """
        Adds a value to single particle electron-phonon couplings
        and correspondingly redefines many-body matrix Vbbp.

        Parameters
        ----------
        velph : dict
            Dictionary describing what values to add.
            For example, velph[(bath, state1, state2)] = value to add.
        updateq : bool
            Specifies if the values of the single particle couplings will be updated.
            The many-body couplings Vbbp will be updates in either case.
        lstq : bool
            Determines if the values will be added to tlst, dlst.
        """
        if lstq:
            self.tlst = self.tlst if tlst is None else self.tlst + make_array(tlst, self.stateind.nleads)
            self.dlst = self.dlst if dlst is None else self.dlst + make_array_dlst(dlst, self.stateind.nleads)
        if not velph is None:
            velphp = velph if isinstance(velph, dict) else make_tleads_dict(velph)
            self.Vbbp0 = elph_construct_Vbbp(velphp, self.stateind, self.mtype, self.Vbbp0)
            if updateq:
                for j0 in velph:
                    try:    self.velph[j0] += velphp[j0]       # if velph[j0] != 0:
                    except: self.velph.update({j0:velphp[j0]}) # if velph[j0] != 0:

    def change(self, velph=None, tlst=None, dlst=None, updateq=True):
        """
        Changes the values of the single particle electron-phonon couplings
        and correspondingly redefines many-body coupling matrix Vbbp.

        Parameters
        ----------
        velph : dict
            Dictionary describing which electron-phonon couplings to change.
            For example, velph[(bath, state1, state2)] = the new value.
        updateq : bool
            Specifies if the values of the single particle couplings will be updated.
            The many-body tunneling amplitudes Vbbp will be updates in either case.
        """
        if not tlst is None:
            if isinstance(tlst, dict):
                for j0 in tlst:
                    self.tlst[j0] = tlst[j0]
            else:
                self.tlst = make_array(tlst, self.stateind.nleads)
        #
        if not dlst is None:
            if isinstance(dlst, dict):
                for j0 in dlst:
                    if isinstance(dlst[j0], numbers.Number):
                        self.dlst[j0] = (-dlst[j0], dlst[j0])
                    else:
                        self.dlst[j0] = dlst[j0]
            else:
                self.dlst = make_array_dlst(dlst, self.stateind.nleads)
        #
        if not velph is None:
            velphp = velph if isinstance(velph, dict) else make_tleads_dict(velph)
            # Find the differences from the previous electron-phonon coupling
            velph_add = {}
            for j0 in velphp:
                try:
                    velph_diff = velphp[j0]-self.velph[j0]
                    if velph_diff != 0:
                        velph_add.update({j0:velph_diff})
                        if updateq: self.velph[j0] += velph_diff
                except:
                    velph_diff = velphp[j0]
                    if velph_diff != 0:
                        velph_add.update({j0:velph_diff})
                        if updateq: self.velph.update({j0:velph_diff})
            # Add the differences
            self.add(velph_add, updateq=False, lstq=False)

    def rotate(self, vecslst, indexing='n'):
        """
        Rotates electron-phonon coupling matrix Vbbp0 in Fock basis to Vbbp,
        which is in eigenstate basis of the quantum dot.

        Parameters
        ----------
        veclst : list of arrays
            List of size ncharge containing arrays defining eigenvector matrices for given charge.
        indexing : string
            Specifies what kind of rotation procedure to use. Default is stateind.indexing.
        """
        self.Vbbp = elph_rotate_Vbbp(self.Vbbp0, vecslst, self.stateind, indexing, self.mtype)

    def use_Vbbp0(self):
        """
        Sets the Vbbp matrix for calculation to Vbbp0 in the Fock basis.
        """
        self.Vbbp = self.Vbbp0

    def update_Vbbp0(self, nbaths, velph, mtype=complex):
        """
        Updates the Vbbp0 in the Fock basis using new single-particle coupling amplitudes.

        Parameters
        ----------
        velph : array
            The new single-particle electron-phonon couplings. See attribute velph.
        """
        self.stateind.nbaths = nbaths
        self.velph = make_velph_dict(velph)
        self.mtype = mtype
        self.Vbbp0 = elph_construct_Vbbp(velph, self.stateind, mtype)