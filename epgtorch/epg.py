# Pytorch EPG Simulation code
# MIT License
# felix.zimmermann@ptb.de
# based off of Matlab scripts by Matthias Weigel
# https://github.com/matthias-weigel/EPG
# and mostly python scripts by Jonathan Tamir
# https://github.com/utcsilab/mri-sim-py

#%%
import torch
from typing import Iterable, Optional


def rf(FpFmZ, matrix):
    """
    Propagate EPG states through an RF rotation
    """
    return torch.matmul(matrix, FpFmZ)


def rf_mat(alphas, phis, B1=None):
    alphas, phis = alphas.moveaxis(-1, 0), phis.moveaxis(-1, 0)
    if B1 is not None:
        alphas = alphas * B1[None, ...]
    cosa = torch.cos(alphas)
    sina = torch.sin(alphas)
    cosa2 = (cosa + 1) / 2
    sina2 = 1 - cosa2

    ejp = torch.exp(1j * phis)
    inv_ejp = 1 / ejp

    RR = torch.stack(
        [cosa2 + 0j, ejp ** 2 * sina2, -1j * ejp * sina, inv_ejp ** 2 * sina2, cosa2 + 0j, 1j * inv_ejp * sina, -1j / 2.0 * inv_ejp * sina, 1j / 2.0 * ejp * sina, cosa + 0j,], -1,
    ).reshape(*alphas.shape, 3, 3)
    return RR


def relax_mat(T, T1, T2):
    E2 = torch.exp(-T / T2)
    E1 = torch.exp(-T / T1)
    E1, E2 = torch.broadcast_tensors(E1, E2)
    mat = torch.stack([E2, E2, E1], dim=-1)

    return mat


def relax(FpFmZ, relax_mat, recovery=True):
    """
    Propagate EPG states through a period of relaxation and recovery
    """
    FpFmZ = relax_mat[..., None] * FpFmZ

    if recovery:
        FpFmZ[..., 2, 0] = FpFmZ[..., 2, 0] + (1 - relax_mat[..., -1])
    return FpFmZ


def grad(FpFmZ, noadd=False):
    """
    Propagate EPG states through a "unit" gradient.
    noadd = True to NOT add any higher-order states - assume
                that they just go to zero.  Be careful - this
                speeds up simulations, but may compromise accuracy!
    """
    zero = torch.zeros(*FpFmZ.shape[:-2], 1, device=FpFmZ.device, dtype=FpFmZ.dtype)
    if noadd:
        Fp = torch.cat((FpFmZ[..., 1, 1:2].conj() if FpFmZ.shape[-1] > 1 else zero, FpFmZ[..., 0, :-1]), -1)
        Fm = torch.cat((FpFmZ[..., 1, 1:], zero), -1)
        Z = FpFmZ[..., 2, :]
    else:
        Fp = torch.cat((FpFmZ[..., 1, 1:2].conj() if FpFmZ.shape[-1] > 1 else zero, FpFmZ[..., 0, :]), -1)
        Fm = torch.cat((FpFmZ[..., 1, 1:], zero, zero), -1)
        Z = torch.cat((FpFmZ[..., 2, :], zero), -1)
    return torch.stack((Fp, Fm, Z), -2)


def FSE_TE(FpFmZ, rf_matrix, EE, recovery=True, noadd=False):
    """Propagate EPG states through a full TE, i.e.
    relax -> grad -> rf -> grad -> relax
    """
    FpFmZ = relax(FpFmZ, EE, recovery)
    FpFmZ = grad(FpFmZ, noadd)
    FpFmZ = rf(FpFmZ, rf_matrix)
    FpFmZ = grad(FpFmZ, noadd)
    FpFmZ = relax(FpFmZ, EE, recovery)

    return FpFmZ


def MRF_TR(FpFmZ, rf_matrix, EETE, EETR, noadd=False):
    """Propagate EPG states through a full MRF TR after an readout, i.e.
    (readout) -> grad -> decay TR-TE -> rf -> decay TE -> (readout) 
    """
    FpFmZ = grad(FpFmZ, noadd)
    FpFmZ = relax(FpFmZ, EETR, True)
    FpFmZ = rf(FpFmZ, rf_matrix)
    FpFmZ = relax(FpFmZ, EETE, True)
    return FpFmZ


def same_along_last_axis(x):
    flat = x.flatten(0, -2)
    return bool(torch.all(flat == flat[:, :1]))


def FSE_signal(
    flipangles: torch.Tensor,
    flipphases: torch.Tensor,
    TE: torch.Tensor,
    T1: torch.Tensor,
    T2: torch.Tensor,
    ETL: Optional[int] = None,
    M0: Optional[torch.Tensor] = None,
    B1: Optional[torch.Tensor] = None,
    prep: Iterable[dict] = None,
):
    """
    Simulate Fast Spin-Echo sequence with specific flip angle/phase train.

    INPUT:
        flipangles, flipphases: (Batch x ETL) or ETL tensors of flip angles and phases in radians
        TE:  (Batch x ETL) echo times
        ETL: echo train length. defaults to non-1 last dimension of flipangles, flipphases and TE
        T1, T2: (Batch) T1,T2 times
        M0: (Batch x 3) magnetistation before first prep pulse
        B1: (Batch) scaling of flipangle
          prep: Iterable of dictionaries for each preperation (excitation, inversion) pulse with keys angle, phase, tau.
        If both M0 and prep are None, assume CPGM excitation
    OUTPUT:
        magnetization at each echo time
    """
    flipangles, flipphases, TE, T1, T2 = (torch.atleast_1d(i) for i in (flipangles, flipphases, TE, T1, T2))
    shape_pulses = torch.broadcast_shapes(flipangles.shape, flipphases.shape, TE.shape)
    shape_prop = torch.broadcast_shapes(T1.shape, T2.shape)
    shape_common = torch.broadcast_shapes(shape_prop, shape_pulses[:-1])
    batch_sizes = shape_common
    T = shape_pulses[-1]
    if ETL is not None:
        if T > 1 and ETL != T:
            raise ValueError("last dimension of flipangles, phases and TE must either be 1 or ETL")
        else:
            T = ETL
    flipangles, flipphases = torch.broadcast_tensors(flipangles, flipphases)
    M = torch.zeros(*batch_sizes, T, 3, device=flipangles.device)
    P = torch.zeros((*batch_sizes, 3, 1), dtype=torch.cfloat, device=flipangles.device)

    if prep is not None:
        if M0 is None:
            P[..., :, 0] = torch.tensor((0.0, 0, 1.0))
        else:
            P[..., :, 0] = M0
        # apply excitation and inversion
        for pulse in prep:
            P = rf(P, rf_mat(pulse["angle"], pulse["phase"], B1))
            if "grad" in pulse and pulse["grad"]:
                P = grad(P, noadd=False)
            if "tau" in pulse:
                P = relax(P, relax_mat(pulse["tau"], T1, T2))
    elif M0 is not None:
        P[..., :, 0] = M0
    else:  # CPMG Excitation
        P[..., :, 0] = torch.tensor((1.0, 1.0, 0.0))
    for i in range(T):
        if i == 0 or flipangles.shape[-1] > 1:
            rf_matrix = rf_mat(flipangles[..., i], flipphases[..., i], B1)
        if i == 0 or TE.shape[-1] > 1:
            relax_matrix = relax_mat(TE[..., i] / 2, T1, T2)
        P = FSE_TE(P, rf_matrix, relax_matrix)
        M[..., i, :] = torch.stack((P[..., 0, 0].real, P[..., 0, 0].imag, P[..., 2, 0].real), -1)
    return M


def MRF_Signal(
    flipangles: torch.Tensor, flipphases: torch.Tensor, TE: torch.Tensor, TR: torch.Tensor, T1: torch.Tensor, T2: torch.Tensor, TI: torch.Tensor,
):
    flipangles, flipphases, TE, TR, T1, T2 = (torch.atleast_1d(i) for i in (flipangles, flipphases, TE, TR, T1, T2))
    flipangles, flipphases = torch.broadcast_tensors(flipangles, flipphases)
    shape_pulses = torch.broadcast_shapes(flipangles.shape, flipphases.shape, TE.shape, TR.shape)
    shape_prop = torch.broadcast_shapes(T1.shape, T2.shape)
    shape_common = torch.broadcast_shapes(shape_prop, shape_pulses[:-1])
    batch_sizes = shape_common
    T = shape_pulses[-1]

    M = torch.zeros(*batch_sizes, T, dtype=torch.cfloat, device=flipangles.device)
    P = torch.zeros((*batch_sizes, 3, 1), dtype=torch.cfloat, device=flipangles.device)

    P[..., :, 0] = torch.tensor((0.0, 0, -1.0))
    P = relax(P, relax_mat(TI, T1, T2))
    for i in range(0, T):
        if i == 0 or flipangles.shape[-1] > 1:
            rf_matrix = rf_mat(flipangles[..., i], flipphases[..., i])
        if i == 0 or TE.shape[-1] > 1:
            TErelax_matrix = relax_mat(TE[..., i], T1, T2)
        if i == 0 or TE.shape[-1] > 1 or TR.shape[-1] > 1:
            TRrelax_matrix = relax_mat((TR - TE)[..., i], T1, T2)

        P = rf(P, rf_matrix)
        P = relax(P, TErelax_matrix, True)
        M[..., i] = P[..., 0, 0]
        P = grad(P, False)
        P = relax(P, TRrelax_matrix, True)
    return M


#%%
if __name__ == "__main__":
    from math import pi

    t1 = torch.tensor(1000.0).requires_grad_(True)
    t2 = torch.tensor(100.0).requires_grad_(True)
    te = torch.tensor(100.0).requires_grad_(True)
    angles = (torch.tile(torch.tensor(120), (6,)) * (pi / 180)).requires_grad_(True)
    phases = torch.zeros(1).requires_grad_(True)
    m = FSE_signal(angles, phases, te, t1, t2)
    echo_intensity = torch.norm(m[..., :2], dim=-1)
    print(echo_intensity.detach())
    echo_intensity.sum().backward()
    print(angles.grad)
