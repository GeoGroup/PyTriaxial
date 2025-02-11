# $$CC391
# -*- coding: UTF-8 -*-
"""
@Project ：PyTriaxial
@File    ：Mohr-Coulomb.py
@IDE     ：PyCharm 
@Author  ：Arjie.
@Date    ：2025-02-06 18:16 
"""

import numpy as np
import math
import matplotlib.pyplot as plt


# =============================================================================
# Main UMAT function
# =============================================================================
def UMAT(STRESS, STATEV, DDSDDE, SSE, SPD, SCD,
         RPL, DDSDDT, DRPLDE, DRPLDT, STRAN, DSTRAN,
         TIME, DTIME, TEMP, DTEMP, PREDEF, DPRED, CMNAME,
         NDI, NSHR, NTENS, NSTATV, PROPS, NPROPS, COORDS,
         DROT, PNEWDT, CELENT, DFGRD0, DFGRD1, NOEL, NPT,
         LAYER, KSPT, KSTEP, KINC):
    """
    Main UMAT function to perform stress update and consistent tangent stiffness matrix generation for Mohr-Coulomb elastoplastic material.

    Input parameters (all NumPy arrays or scalars) are the same as those in the Abaqus UMAT interface,
    where STRESS, STATEV, DDSDDE, etc., will be updated within this subroutine.

    Parameters explanation (partial):
      - STRESS: Current stress vector
      - DSTRAN: Strain increment
      - PROPS: Material property array, in order [E, nu, coh, phi_d, psi_d, …]
      - NTENS: Number of stress/strain components (e.g., 4 or 6)
      - KINC, NOEL, NPT, etc., are used for output debugging information

    Return:
      Updated STRESS, STATEV, DDSDDE (other variables remain unchanged)
    """
    # Material properties
    E = PROPS[0]  # Young's modulus
    nu = PROPS[1]  # Poisson's ratio
    coh = PROPS[2]  # Cohesion
    phi_d = PROPS[3]  # Friction angle (in degrees)
    psi_d = PROPS[4]  # Dilation angle (in degrees)
    pi_val = math.pi

    # Convert angles from degrees to radians
    phi = pi_val * phi_d / 180.0
    psi = pi_val * psi_d / 180.0

    # -----------------------------------------------------------------------
    # Calculate the elastic stiffness matrix D and its inverse Dinv
    # -----------------------------------------------------------------------
    D, Dinv = DlinElas(E, nu, NTENS)

    # -----------------------------------------------------------------------
    # Elastic stress prediction: DSTRES = D * DSTRAN, SigB = STRESS + DSTRES
    # -----------------------------------------------------------------------
    DSTRES = np.matmul(D, DSTRAN)
    SigB = STRESS + DSTRES

    PlastPar = np.zeros(3, dtype=np.float64)
    PlastPar[0] = (1.0 + math.sin(phi)) / (1.0 - math.sin(phi))
    PlastPar[1] = 2.0 * coh * math.sqrt(PlastPar[0])
    PlastPar[2] = (1.0 + math.sin(psi)) / (1.0 - math.sin(psi))

    # -----------------------------------------------------------------------
    # Stress return mapping and consistent tangent stiffness matrix generation
    # -----------------------------------------------------------------------
    SigC, Depc, region = MohrCoulombStressReturn(SigB, NTENS, PlastPar, D, Dinv)

    # ABAQUS post-processor: Update the consistent tangent stiffness matrix and stress
    DDSDDE[:] = Depc  # Consistent tangent stiffness matrix
    STRESS[:] = SigC  # Updated stress

    # Update state variables STATEV: Store region number in STATEV[0]
    STATEV[0] = float(region)

    # Return updated main variables (other variables remain the same)
    return STRESS, STATEV, DDSDDE


# =============================================================================
# Auxiliary functions
# =============================================================================

def DlinElas(E, nu, nsigma):
    """
    Calculate the linear elastic stiffness matrix D and its inverse Dinv.

    Based on the value of nsigma:
      - nsigma = 3: Plane stress problem
      - nsigma = 4: Plane strain or axisymmetric problem
      - nsigma = 6: Three-dimensional stress state

    Return:
      D, Dinv are (nsigma x nsigma) NumPy arrays.
    """
    if nsigma == 3:
        D = np.zeros((3, 3), dtype=np.float64)
        D[0, 0] = 1.0;
        D[0, 1] = nu
        D[1, 0] = nu;
        D[1, 1] = 1.0
        D[2, 2] = (1.0 - nu) / 2.0
        D = E / (1.0 - nu * nu) * D

        Dinv = np.zeros((3, 3), dtype=np.float64)
        Dinv[0, 0] = 1.0;
        Dinv[0, 1] = -nu
        Dinv[1, 0] = -nu;
        Dinv[1, 1] = 1.0
        Dinv[2, 2] = 2.0 * (1.0 + nu)
        Dinv = Dinv / E
        return D, Dinv

    elif nsigma == 4:
        D = np.zeros((4, 4), dtype=np.float64)
        D[0, 0] = 1.0 - nu;
        D[0, 1] = nu;
        D[0, 2] = nu
        D[1, 0] = nu;
        D[1, 1] = 1.0 - nu;
        D[1, 2] = nu
        D[2, 0] = nu;
        D[2, 1] = nu;
        D[2, 2] = 1.0 - nu
        D[3, 3] = (1.0 - 2.0 * nu) / 2.0
        D = E / ((1.0 + nu) * (1.0 - 2.0 * nu)) * D

        Dinv = np.zeros((4, 4), dtype=np.float64)
        Dinv[0, 0] = 1.0;
        Dinv[0, 1] = -nu;
        Dinv[0, 2] = -nu
        Dinv[1, 0] = -nu;
        Dinv[1, 1] = 1.0;
        Dinv[1, 2] = -nu
        Dinv[2, 0] = -nu;
        Dinv[2, 1] = -nu;
        Dinv[2, 2] = 1.0
        Dinv[3, 3] = 2.0 * (1.0 + nu)
        Dinv = Dinv / E
        return D, Dinv

    elif nsigma == 6:
        D = np.zeros((6, 6), dtype=np.float64)
        c = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
        g = E / (2.0 * (1.0 + nu))
        D[0, 0] = (1.0 - nu) * c;
        D[0, 1] = nu * c;
        D[0, 2] = nu * c
        D[1, 0] = nu * c;
        D[1, 1] = (1.0 - nu) * c;
        D[1, 2] = nu * c
        D[2, 0] = nu * c;
        D[2, 1] = nu * c;
        D[2, 2] = (1.0 - nu) * c
        D[3, 3] = g;
        D[4, 4] = g;
        D[5, 5] = g

        Dinv = np.zeros((6, 6), dtype=np.float64)
        Dinv[0, 0] = 1.0;
        Dinv[0, 1] = -nu;
        Dinv[0, 2] = -nu
        Dinv[1, 0] = -nu;
        Dinv[1, 1] = 1.0;
        Dinv[1, 2] = -nu
        Dinv[2, 0] = -nu;
        Dinv[2, 1] = -nu;
        Dinv[2, 2] = 1.0
        Dinv[3, 3] = 2.0 * (1.0 + nu);
        Dinv[4, 4] = 2.0 * (1.0 + nu);
        Dinv[5, 5] = 2.0 * (1.0 + nu)
        Dinv = Dinv / E
        return D, Dinv

    else:
        raise ValueError("Unsupported nsigma value.")


# ---------------------------------------------------------------------------
def MohrCoulombStressReturn(Sigma, nsigma, PlasPar, D, Dinv):
    """
    Stress return mapping subroutine for a linear elastic-perfectly plastic Mohr-Coulomb material model.
    Calculations are done in the principal stress space.

    Input:
      - Sigma: Predicted stress vector
      - nsigma: Number of stress components (4 or 6)
      - PlasPar: Plastic material parameters array [k, comp, m]
      - D, Dinv: Elastic stiffness matrix and its inverse

    Output:
      - Sigma_up: Updated global stress vector
      - Depc: Consistent tangent stiffness matrix (global coordinates)
      - region: Stress return region number (0 indicates elasticity)
    """
    # Compute principal stresses (using PrinStressAna)
    SigP = PrinStressAna(Sigma, nsigma)

    # Yield function f = k * sigP_1 - sigP_3 - comp
    k_val = PlasPar[0]
    comp = PlasPar[1]
    f = k_val * SigP[0] - SigP[2] - comp

    if f > 0.0:
        # For nsigma==4, determine the out-of-plane principal stress location
        if nsigma == 4:
            if np.isclose(Sigma[2], SigP[0]):
                ouplP = 1;
                s1 = 2;
                s2 = 3
            elif np.isclose(Sigma[2], SigP[1]):
                ouplP = 2;
                s1 = 1;
                s2 = 3
            elif np.isclose(Sigma[2], SigP[2]):
                ouplP = 3;
                s1 = 1;
                s2 = 2
            else:
                ouplP = 1;
                s1 = 2;
                s2 = 3  # Default value
        else:
            ouplP = None  # Do not use for nsigma != 4

        # Stress return in principal stress space (using PrinRetMoCo)
        SigP_up, region = PrinRetMoCo(SigP, f, PlasPar, D, nsigma)
        # Plastic correction vector SiPla = SigP - SigP_up
        SiPla = SigP - SigP_up

        # Number of shear stress components: nshear = nsigma - 3
        nshear = nsigma - 3
        # For plane problems, calculate shear part correction matrix Tshear (using TshearPrinPerfect)
        if nsigma == 4:
            Tshear, Tshearinv = TshearPrinPerfect(SigP, SigP_up, nshear, s1, s2)
        else:
            Tshear = None
            Tshearinv = None

        # Construct correction matrix T in principal stress space: First 3×3 is identity matrix, remaining part replaced by Tshear
        T = np.eye(nsigma)
        if nsigma > 3 and Tshear is not None:
            T[3:nsigma, 3:nsigma] = Tshear

        # Calculate infinitesimal tangent stiffness matrix DepP in principal stress space
        DepP = np.zeros((nsigma, nsigma), dtype=np.float64)
        # Parameters for modifying DepP
        DepFormLine = 1  # 1: Modified "near double singular matrix" form
        beta = 30.0
        DepFormApex = 1  # 1: Single singular matrix form (vertex region)
        alpha = 70.0

        if region == 1:
            # Return to yield surface: Shear part uses D's shear components
            if nsigma > 3:
                DepP[3:nsigma, 3:nsigma] = D[3:nsigma, 3:nsigma]
            Fnorm = np.array([PlasPar[0], 0.0, -1.0], dtype=np.float64)
            Gnorm = np.array([PlasPar[2], 0.0, -1.0], dtype=np.float64)
            DepP[0:3, 0:3] = FormDepPerfect(D[0:3, 0:3], Fnorm, Gnorm, 3)
        elif region == 2:
            # Return to triaxial compression line (sigp1 = sigp2)
            Lfdir = np.array([1.0, 1.0, PlasPar[0]], dtype=np.float64)
            Lgdir = np.array([1.0, 1.0, PlasPar[2]], dtype=np.float64)
            DepP = FormModDepLine(DepFormLine, beta, SiPla, Lfdir, Lgdir,
                                  Dinv[0:3, 0:3], D, Dinv, nsigma)
        elif region == 3:
            # Return to triaxial tension line (sigp2 = sigp3)
            Lfdir = np.array([1.0, PlasPar[0], PlasPar[0]], dtype=np.float64)
            Lgdir = np.array([1.0, PlasPar[2], PlasPar[2]], dtype=np.float64)
            DepP = FormModDepLine(DepFormLine, beta, SiPla, Lfdir, Lgdir,
                                  Dinv[0:3, 0:3], D, Dinv, nsigma)
        else:
            # Return to vertex
            DepP = FormModDepApex(DepFormApex, alpha, SiPla,
                                  Dinv[0:3, 0:3], D, Dinv, nsigma)

        # Combine correction matrix T and DepP in principal stress space
        DepcP = np.matmul(T, DepP)

        # Calculate transformation matrix A from principal stress coordinates to global coordinates
        psi = PrinDirect(Sigma, nsigma, SigP)
        A = TransMatrix(psi, nsigma, ouplP) if nsigma == 4 else np.eye(nsigma)
        Atrans = A.T

        # Coordinate transformation: Global updated stress Sigma_up = Atrans[:,0:3] * SigP_up
        Sigma_up = np.matmul(Atrans[:, 0:3], SigP_up)
        # Global consistent tangent stiffness matrix: Depc = Atrans * DepcP * A
        Depc = np.matmul(np.matmul(Atrans, DepcP), A)
    else:
        # No yield: Directly return original stress and elastic stiffness matrix
        Sigma_up = Sigma.copy()
        Depc = D.copy()
        region = 0

    return Sigma_up, Depc, region


# ---------------------------------------------------------------------------
def PrinRetMoCo(SigP, f, PlasPar, Dfull, nsigma):
    """
    Calculate the stress return in the principal stress space for Mohr-Coulomb plastic model,
    considering non-associated flow but ignoring hardening behavior.

    Input:
      - SigP: Predicted principal stress array [sigP_1, sigP_2, sigP_3] (in descending order)
      - f: Initial yield function value f = k * sigP_1 - sigP_3 - comp
      - PlasPar: Plastic material parameters array [k, comp, m]
      - Dfull: Complete elastic stiffness matrix (nsigma x nsigma)
      - nsigma: Number of stress components

    Output:
      - SigP_up: Updated principal stress array
      - region: Stress return region number
         region = 1: Return to yield surface
         region = 2: Return to triaxial compression line
         region = 3: Return to triaxial tension line
         region = 4: Return to vertex
    """
    k_val = PlasPar[0]
    m = PlasPar[2]
    comp = PlasPar[1]
    # Compute apex position: apex = comp / (k - 1)
    apex = comp / (k_val - 1.0)
    # Take the first 3×3 part of Dfull (stiffness matrix related to principal stress direction)
    D_mat = Dfull[0:3, 0:3]

    # Compute Rp = D*b/(a'*D*b), where a and b are gradients of yield surface and plastic potential surface
    den = k_val * (D_mat[0, 0] * m - D_mat[0, 2]) - D_mat[2, 0] * m + D_mat[2, 2]
    Rp = np.zeros(3, dtype=np.float64)
    Rp[0] = (D_mat[0, 0] * m - D_mat[0, 2]) / den
    Rp[1] = (D_mat[1, 0] * m - D_mat[1, 2]) / den
    Rp[2] = (D_mat[2, 0] * m - D_mat[2, 2]) / den

    # From predicted principal stresses to apex vector: SigPapex = SigP - apex (scalar minus vector, component-wise)
    SigPapex = SigP - apex

    # Boundary plane (region I and II) normal vector: NI_II = [Rp[1]*k - Rp[2], Rp[2] - Rp[0]*k, Rp[0] - Rp[1]]
    # Boundary plane (region I and II) normal vector: NI_II = [Rp[1]*k - Rp[2], Rp[2] - Rp[0]*k, Rp[0] - Rp[1]]
    NI_II = np.array([Rp[1] * k_val - Rp[2], Rp[2] - Rp[0] * k_val, Rp[0] - Rp[1]], dtype=np.float64)
    pI_II = np.dot(NI_II, SigPapex)

    # Boundary plane (region I and III) normal vector: NI_III = [Rp[1]*k - Rp[2]*k, Rp[2] - Rp[0]*k, Rp[0]*k - Rp[1]]
    NI_III = np.array([Rp[1] * k_val - Rp[2] * k_val, Rp[2] - Rp[0] * k_val, Rp[0] * k_val - Rp[1]], dtype=np.float64)
    pI_III = np.dot(NI_III, SigPapex)

    # Calculate the t parameter for region II: using secondary boundary surface a = [0, k, -1]
    den_temp = k_val * (D_mat[1, 1] * m - D_mat[1, 2]) - D_mat[2, 1] * m + D_mat[2, 2]
    Rp2 = np.zeros(3, dtype=np.float64)
    Rp2[0] = (D_mat[0, 1] * m - D_mat[0, 2]) / den_temp
    Rp2[1] = (D_mat[1, 1] * m - D_mat[1, 2]) / den_temp
    Rp2[2] = (D_mat[2, 1] * m - D_mat[2, 2]) / den_temp
    N2 = np.cross(Rp, Rp2)
    den_t = N2[0] + N2[1] + k_val * N2[2]
    t1 = np.dot(N2, SigPapex) / den_t if den_t != 0 else 0.0

    # Calculate the t parameter for region III: using secondary boundary surface a = [k, -1, 0]
    den_temp = k_val * (D_mat[0, 0] * m - D_mat[0, 1]) - D_mat[1, 0] * m + D_mat[1, 1]
    Rp3 = np.zeros(3, dtype=np.float64)
    Rp3[0] = (D_mat[0, 0] * m - D_mat[0, 1]) / den_temp
    Rp3[1] = (D_mat[1, 0] * m - D_mat[1, 1]) / den_temp
    Rp3[2] = (D_mat[2, 0] * m - D_mat[2, 1]) / den_temp
    N3 = np.cross(Rp, Rp3)
    den_t = N3[0] + k_val * N3[1] + k_val * N3[2]
    t2 = np.dot(N3, SigPapex) / den_t if den_t != 0 else 0.0

    # Determine the stress return region based on t1, t2, and the boundary plane projection values
    SigP_up = np.zeros(3, dtype=np.float64)
    if t1 > 0.0 and t2 > 0.0:
        region = 4
        SigP_up[:] = apex
    elif pI_II < 0.0:
        region = 2
        SigP_up[0] = t1 + apex
        SigP_up[1] = t1 + apex
        SigP_up[2] = t1 * k_val + apex
    elif pI_III <= 0.0:
        region = 1
        SigP_up = SigP - f * Rp
    else:
        region = 3
        SigP_up[0] = t2 + apex
        SigP_up[1] = t2 * k_val + apex
        SigP_up[2] = t2 * k_val + apex

    return SigP_up, region


# ---------------------------------------------------------------------------
def FormModDepLine(mod_type, beta, SiPla, Fline, Gline, Dninv, Dc, Dcinv, nsigma):
    """
    Calculate the modified elastic-plastic tangent stiffness matrix on the boundary line.
    Input parameters:
      - mod_type: Modification type (0 or 1 or 2)
      - beta: Modification factor
      - SiPla: Plastic correction vector (3D)
      - Fline, Gline: Yield boundary line and plastic potential boundary line directions (3D vectors)
      - Dninv: Elastic compliance matrix in the principal stress directions (3×3)
      - Dc, Dcinv: Modified stiffness matrix and its inverse (nsigma×nsigma)
      - nsigma: Number of stress components
    Returns:
      - Depc: Modified tangent stiffness matrix
    """
    # Call FormDepLinePerfect to calculate the theoretically predicted double singular matrix
    Depc = FormDepLinePerfect(Dc, Dcinv, Fline, Gline, nsigma)
    if mod_type == 1 and np.dot(SiPla, SiPla) > 0.0:
        KoitDir = np.matmul(Dninv, SiPla)  # Plastic strain direction
        KoitPerDir = Cross(KoitDir,
                           Gline)  # Direction perpendicular to both plastic strain and plastic potential directions
        Dper = FormDepLinePerfect(Dc[0:3, 0:3], Dcinv[0:3, 0:3], KoitPerDir, KoitPerDir, 3)
        Depc[0:3, 0:3] += Dper / beta
    elif mod_type == 2 and np.dot(SiPla, SiPla) > 0.0:
        KoitDir = np.matmul(Dninv, SiPla)
        Dep_norm = FormDepPerfect(Dc[0:3, 0:3], KoitDir, KoitDir, 3)
        Depc[0:3, 0:3] = Dep_norm
        if nsigma > 3:
            Depc[3:nsigma, 3:nsigma] = Dc[3:nsigma, 3:nsigma]
    return Depc


# ---------------------------------------------------------------------------
def FormModDepApex(mod_type, alpha, SiPla, Dninv, Dc, Dcinv, nsigma):
    """
    Calculate the modified elastic-plastic tangent stiffness matrix at the apex.
    Input parameters are similar to FormModDepLine, mod_type determines the specific modification method:
      - mod_type = 0: Use zero matrix (not used here)
      - mod_type = 1: Single singular matrix, plastic strain direction is singular
      - mod_type = 2: Double singular matrix, both plastic strain and hydrostatic stress directions are singular
    Returns:
      - Depc: Modified tangent stiffness matrix (nsigma×nsigma)
    """
    Depc = np.zeros((nsigma, nsigma), dtype=np.float64)
    if np.dot(SiPla, SiPla) > 0.0:
        KoitDir = np.matmul(Dninv, SiPla)
        if mod_type == 1:
            Dnorm = FormDepPerfect(Dc[0:3, 0:3], KoitDir, KoitDir, 3)
            Depc[0:3, 0:3] = Dnorm / alpha
            if nsigma > 3:
                Depc[3:nsigma, 3:nsigma] = Dc[3:nsigma, 3:nsigma] / alpha
        elif mod_type == 2:
            Pdir = np.array([1.0, 1.0, 1.0], dtype=np.float64)
            PerDir = Cross(KoitDir, Pdir)
            Depc = FormDepLinePerfect(Dc, Dcinv, PerDir, PerDir, nsigma)
            Depc = Depc / alpha
    return Depc


# ---------------------------------------------------------------------------
def TshearPrinPerfect(SigP, SigP_up, nshear, s1, s2):
    """
    Calculate the shear correction matrix T and its inverse matrix in principal stress space.
    For plane stress problems (nshear = 1 or 3), s1 and s2 represent the locations of the maximum and minimum principal in-plane stresses.

    Returns:
      - Tshear, Tshearinv: Shear correction matrix and its inverse matrix (nshear×nshear)
    """
    Tshear = np.zeros((nshear, nshear), dtype=np.float64)
    Tshearinv = np.zeros((nshear, nshear), dtype=np.float64)

    if nshear == 1:
        idx1 = s1 - 1  # Adjust to 0 index
        idx2 = s2 - 1
        if (SigP_up[idx1] - SigP_up[idx2]) > 0.0:
            Ts = (SigP_up[idx1] - SigP_up[idx2]) / (SigP[idx1] - SigP[idx2])
            Tshear[0, 0] = Ts
            Tshearinv[0, 0] = 1.0 / Ts
    elif nshear == 3:
        if (SigP_up[0] - SigP_up[1] > 0.0) and (SigP[0] - SigP[1] > 0.0):
            Tshear[0, 0] = (SigP_up[0] - SigP_up[1]) / (SigP[0] - SigP[1])
            Tshearinv[0, 0] = 1.0 / Tshear[0, 0]
        if (SigP_up[0] - SigP_up[2] > 0.0) and (SigP[0] - SigP[2] > 0.0):
            Tshear[1, 1] = (SigP_up[0] - SigP_up[2]) / (SigP[0] - SigP[2])
            Tshearinv[1, 1] = 1.0 / Tshear[1, 1]
        if (SigP_up[1] - SigP_up[2] > 0.0) and (SigP[1] - SigP[2] > 0.0):
            Tshear[2, 2] = (SigP_up[1] - SigP_up[2]) / (SigP[1] - SigP[2])
            Tshearinv[2, 2] = 1.0 / Tshear[2, 2]
    return Tshear, Tshearinv


# ---------------------------------------------------------------------------
def PrinStressAna(Sigma, nsigma):
    """
    Calculate the principal stresses of the stress vector Sigma.
    For plane problems (nsigma = 4) and three-dimensional stress states (nsigma = 6), respectively handle,
    Returns an array containing 3 principal stresses (sorted in descending order).

    Input:
      - Sigma: Stress vector
      - nsigma: Number of stress components

    Returns:
      - SigP: Principal stress array (3,)
    """
    SigP = np.zeros(3, dtype=np.float64)
    if nsigma == 4:
        sig_av = 0.5 * (Sigma[0] + Sigma[1])
        sig_hj = math.sqrt((0.5 * (Sigma[0] - Sigma[1])) ** 2 + Sigma[3] ** 2)
        SigP[0] = sig_av + sig_hj
        SigP[1] = sig_av - sig_hj
        SigP[2] = Sigma[2]  # Out-of-plane principal stress
        # Sorting (in descending order)
        SigP = np.sort(SigP)[::-1]
    elif nsigma == 6:
        # Calculate stress invariants (call Invariants)
        S, I1, J2, J3, lode, sin3lode = Invariants(Sigma, nsigma)
        sigm = I1 / 3.0
        if math.sqrt(J2) < 1.0e-12 * abs(I1):
            SigP[0] = Sigma[0]
            SigP[1] = Sigma[1]
            SigP[2] = Sigma[2]
        else:
            sqJ2 = 1.154700538379252 * math.sqrt(J2)  # 2/sqrt(3)
            SigP[0] = sqJ2 * math.sin(lode + 2.094395102393195) + sigm
            SigP[1] = sqJ2 * math.sin(lode) + sigm
            SigP[2] = sqJ2 * math.sin(lode - 2.094395102393195) + sigm
    return SigP


# ---------------------------------------------------------------------------
def FormDepPerfect(D, Norm, Edir, nsigma):
    """
    Calculate the elastic-plastic constitutive matrix for perfectly plastic material with non-associative behavior.
    Input:
      - D: Elastic constitutive matrix (nsigma×nsigma)
      - Norm: Yield surface normal vector (nsigma)
      - Edir: Plastic potential normal vector (nsigma)
    Returns:
      - Dep: Consistent tangent stiffness matrix
    """
    Num1 = np.matmul(D, Edir)
    Num2 = np.matmul(Norm, D)  # Get the 1D array
    Num = np.outer(Num1, Num2)
    den = np.dot(Norm, Num1)
    Dep = D - Num / den
    return Dep


# ---------------------------------------------------------------------------
def FormDepLinePerfect(D, Dinv, Ra, Rb, nsigma):
    """
    Calculate the double singular elastic-plastic constitutive matrix for perfectly plastic material with non-associative behavior (in principal stress space).
    Input:
      - D: Elastic constitutive matrix (nsigma×nsigma)
      - Dinv: Inverse of D matrix
      - Ra: Tangent direction on the yield surface (3D vector)
      - Rb: Tangent direction on the plastic potential surface (3D vector)
    Returns:
      - DepLine: Double singular elastic-plastic constitutive matrix (nsigma×nsigma)
    """
    Num = np.outer(Ra, Rb)
    Den1 = np.matmul(Dinv[0:3, 0:3], Rb)
    den = np.dot(Ra, Den1)
    Dprin = Num / den
    DepLine = np.zeros((nsigma, nsigma), dtype=np.float64)
    DepLine[0:3, 0:3] = Dprin
    if nsigma > 3:
        DepLine[3, 3] = D[3, 3]  # Plane problem when nsigma==4
        if nsigma == 6:
            DepLine[4, 4] = D[4, 4]
            DepLine[5, 5] = D[5, 5]
    return DepLine


# ---------------------------------------------------------------------------
def PrinDirect(Sigma, nsigma, SigP):
    """
    Calculate the principal directions of the principal stress state.
    For plane problems (nsigma = 4), return the rotation angle stored in psi[0,0];
    For three-dimensional stress states (nsigma = 6), return the normalized eigenvector matrix.

    Input:
      - Sigma: Global stress vector
      - nsigma: Number of stress components
      - SigP: Principal stress array (3,)
    Returns:
      - psi: Corresponding principal directions (3×3 matrix or angle)
    """
    psi = np.zeros((3, 3), dtype=np.float64)
    pi_val = math.pi

    if nsigma == 4:
        # Plane stress problem: Calculate the rotation angle
        if Sigma[0] > Sigma[1] and Sigma[3] >= 0.0:
            angle = 0.5 * math.atan2(2 * Sigma[3], (Sigma[0] - Sigma[1]))
        elif Sigma[0] < Sigma[1] and Sigma[3] >= 0.0:
            angle = 0.5 * (pi_val - math.atan2(2 * Sigma[3], (Sigma[1] - Sigma[0])))
        elif Sigma[0] < Sigma[1] and Sigma[3] < 0.0:
            angle = 0.5 * (math.atan2(-2 * Sigma[3], (Sigma[1] - Sigma[0])) + pi_val)
        elif Sigma[0] > Sigma[1] and Sigma[3] < 0.0:
            angle = 0.5 * (2 * pi_val - math.atan2(-2 * Sigma[3], (Sigma[0] - Sigma[1])))
        elif Sigma[0] == Sigma[1] and Sigma[3] > 0.0:
            angle = 0.25 * pi_val
        elif Sigma[0] == Sigma[1] and Sigma[3] < 0.0:
            angle = 0.75 * pi_val
        elif Sigma[0] == Sigma[1] and Sigma[3] == 0.0:
            angle = 0.0
        psi[0, 0] = angle
    elif nsigma == 6:
        # Three-dimensional stress state: Use analytical expressions or numerical methods to solve for principal directions
        tol1 = abs(1.0e-10 * (SigP[0] - SigP[2]))
        tol2 = abs(1.0e-12 * (SigP[0] + SigP[1] + SigP[2]))
        tol = max(tol1, tol2, 1.0e-13)
        normal = np.zeros((3, 3), dtype=np.float64)
        if abs(Sigma[3]) < tol and abs(Sigma[4]) < tol and abs(Sigma[5]) < tol:
            # When stress is already principal stress, take unit matrix
            if Sigma[0] >= Sigma[1] and Sigma[0] >= Sigma[2]:
                normal[:, 0] = [1.0, 0.0, 0.0]
                if Sigma[1] >= Sigma[2]:
                    normal[:, 1] = [0.0, 1.0, 0.0]
                    normal[:, 2] = [0.0, 0.0, 1.0]
                else:
                    normal[:, 1] = [0.0, 0.0, 1.0]
                    normal[:, 2] = [0.0, 1.0, 0.0]
            elif Sigma[1] >= Sigma[0] and Sigma[1] >= Sigma[2]:
                normal[:, 1] = [0.0, 1.0, 0.0]
                if Sigma[0] >= Sigma[2]:
                    normal[:, 0] = [1.0, 0.0, 0.0]
                    normal[:, 2] = [0.0, 0.0, 1.0]
                else:
                    normal[:, 0] = [0.0, 0.0, 1.0]
                    normal[:, 2] = [1.0, 0.0, 0.0]
            elif Sigma[2] >= Sigma[0] and Sigma[2] >= Sigma[1]:
                normal[:, 2] = [0.0, 0.0, 1.0]
                if Sigma[0] >= Sigma[1]:
                    normal[:, 0] = [1.0, 0.0, 0.0]
                    normal[:, 1] = [0.0, 1.0, 0.0]
                else:
                    normal[:, 0] = [0.0, 1.0, 0.0]
                    normal[:, 1] = [1.0, 0.0, 0.0]
        elif (abs(SigP[0] - SigP[1]) > tol and abs(SigP[1] - SigP[2]) > tol and abs(SigP[0] - SigP[2]) > tol):
            # For three different eigenvalues, use numerical methods (e.g., SVD)
            for i in range(2):
                A = np.array([[Sigma[0] - SigP[i], Sigma[3], Sigma[4]],
                              [Sigma[3], Sigma[1] - SigP[i], Sigma[5]],
                              [Sigma[4], Sigma[5], Sigma[2] - SigP[i]]], dtype=np.float64)
                U, s, Vt = np.linalg.svd(A)
                vec = Vt[-1, :]
                normal[:, i] = vec / np.linalg.norm(vec)
            # The third principal direction is the cross product of the first two eigenvectors
            normal[:, 2] = Cross(normal[:, 0], normal[:, 1])
        elif (abs(SigP[0] - SigP[1]) < tol and abs(SigP[1] - SigP[2]) < tol):
            normal = np.eye(3)
        else:
            # When there are two equal eigenvalues, use numpy's eigenvalue decomposition
            stress_tensor = np.array([[Sigma[0], Sigma[3], Sigma[4]],
                                      [Sigma[3], Sigma[1], Sigma[5]],
                                      [Sigma[4], Sigma[5], Sigma[2]]], dtype=np.float64)
            vals, vecs = np.linalg.eig(stress_tensor)
            idx = np.argsort(vals)[::-1]
            normal = vecs[:, idx]
        psi = normal
    return psi


# ---------------------------------------------------------------------------
def Cross(A, B):
    """
    Calculate the cross product (vector product) of two 3D vectors A and B.
    """
    return np.cross(A, B)


# ---------------------------------------------------------------------------
def TransMatrix(psi, nsigma, ouplP):
    """
    Calculate the stress transformation matrix A based on rotation angle (plane stress/strain problem) or direction cosines (three-dimensional stress state).

    Input:
      - psi: When nsigma==4, psi[0,0] is the rotation angle (in radians); when nsigma==6, psi is a 3×3 direction cosine matrix.
      - nsigma: Number of stress components (4 or 6)
      - ouplP: Position of out-of-plane principal stress in plane problems (1, 2, or 3)

    Output:
      - A: Stress transformation matrix (nsigma×nsigma)
    """
    A = np.zeros((nsigma, nsigma), dtype=np.float64)
    if nsigma == 4:
        angle = psi[0, 0]
        sin_psi = math.sin(angle)
        cos_psi = math.cos(angle)
        sin_psi2 = sin_psi ** 2
        cos_psi2 = cos_psi ** 2
        sin_2psi = math.sin(2.0 * angle)

        # Construct the 3×3 Asmall matrix
        Asmall = np.zeros((3, 3), dtype=np.float64)
        Asmall[0, 0] = cos_psi2;
        Asmall[0, 1] = sin_psi2
        Asmall[1, 0] = sin_psi2;
        Asmall[1, 1] = cos_psi2
        Asmall[2, 0] = -sin_2psi;
        Asmall[2, 1] = sin_2psi
        Asmall[0, 2] = 0.5 * sin_2psi
        Asmall[1, 2] = -0.5 * sin_2psi
        Asmall[2, 2] = cos_psi2 - sin_psi2

        # ExtP matrix is used to extend to 4×3, depending on ouplP
        ExtP = np.zeros((4, 3), dtype=int)
        if ouplP == 2:
            ExtP[0, 0] = 1;
            ExtP[2, 1] = 1;
            ExtP[3, 2] = 1
        elif ouplP == 1:
            ExtP[1, 0] = 1;
            ExtP[2, 1] = 1;
            ExtP[3, 2] = 1
        elif ouplP == 3:
            ExtP[0, 0] = 1;
            ExtP[1, 1] = 1;
            ExtP[3, 2] = 1

        # Ext matrix assumes out-of-plane stress is at the 3rd component (index 2)
        Ext = np.zeros((3, 4), dtype=int)
        Ext[0, 0] = 1;
        Ext[1, 1] = 1;
        Ext[2, 3] = 1

        Aint = np.matmul(Asmall, Ext)
        A = np.matmul(ExtP, Aint)
        A[ouplP - 1, 2] = 1.0

    elif nsigma == 6:
        # Construct the transformation matrix for three-dimensional stress state
        A1 = psi ** 2
        A2 = np.zeros((3, 3), dtype=np.float64)
        A3 = np.zeros((3, 3), dtype=np.float64)
        A4 = np.zeros((3, 3), dtype=np.float64)

        Hj1 = np.array([0, 2, 1])
        Hj2 = np.array([1, 0, 2])
        for i in range(3):
            for j in range(3):
                A2[i, j] = psi[i, Hj1[j]] * psi[i, Hj2[j]]
                A3[i, j] = psi[Hj1[i], j] * psi[Hj2[i], j]
                A4[i, j] = psi[Hj1[i], Hj1[j]] * psi[Hj2[i], Hj2[j]] + psi[Hj2[i], Hj1[j]] * psi[Hj1[i], Hj2[j]]
        A2 = 2 * A2
        A[0:3, 0:3] = A1
        A[0:3, 3:6] = A2
        A[3:6, 0:3] = A3
        A[3:6, 3:6] = A4
        A = A.T
    return A


# ---------------------------------------------------------------------------
def Invariants(Sigma, nsigma):
    """
    Calculate the deviatoric stress vector and several stress invariants:
      - I1: First invariant (sum)
      - J2: Second deviatoric stress invariant
      - J3: Third deviatoric stress invariant
      - lode: Lode angle (in radians)
      - sin3lode: Sine of three times the Lode angle

    Input:
      - Sigma: Stress vector (size depends on nsigma)
      - nsigma: Number of stress components (3, 4, or 6)

    Returns:
      - S, I1, J2, J3, lode, sin3lode
    """
    S = np.zeros(nsigma, dtype=np.float64)
    I1 = np.sum(Sigma[0:3])
    if nsigma == 3:
        S[0:3] = Sigma[0:3] - I1 / 3.0
        J2 = 0.5 * np.sum(S[0:3] ** 2)
        J3 = S[0] * S[1] * S[2]
    elif nsigma == 4:
        S[0:3] = Sigma[0:3] - I1 / 3.0
        S[3] = Sigma[3]
        J2 = 0.5 * np.sum(S[0:3] ** 2) + S[3] ** 2
        J3 = (S[0] ** 3 + S[1] ** 3 + S[2] ** 3 + 3.0 * S[3] ** 2 * (S[0] + S[1])) / 3.0
    elif nsigma == 6:
        S[0:3] = Sigma[0:3] - I1 / 3.0
        S[3:6] = Sigma[3:6]
        J2 = 0.5 * np.sum(S[0:3] ** 2) + np.sum(S[3:6] ** 2)
        J3 = (np.sum(S[0:3] ** 3) + 6.0 * S[3] * S[4] * S[5] +
              3.0 * (S[0] * (S[3] ** 2 + S[4] ** 2) +
                     S[1] * (S[3] ** 2 + S[5] ** 2) +
                     S[2] * (S[4] ** 2 + S[5] ** 2))) / 3.0

    if J2 > 0.0:
        sin3lode = -2.598076211353316 * J3 / (J2 ** 1.5)
    else:
        sin3lode = 0.0

    if sin3lode <= -1.0:
        lode = -0.5235987755982988  # -pi/6
    elif sin3lode >= 1.0:
        lode = 0.5235987755982988  # +pi/6
    else:
        lode = math.asin(sin3lode) / 3.0

    return S, I1, J2, J3, lode, sin3lode


# -----------------------------FUNCTION END----------------------------------


if __name__ == "__main__":
    # ================== Input Parameters ======================
    # Rock test parameters
    confining_stress = -1e6  # Confining stress [Pa]
    total_axial_strain = -0.035  # Total axial strain (compression is negative)
    nsteps = 1000  # Number of steps
    d_epsilon_z = total_axial_strain / nsteps

    # Material parameters
    E = 1e8  # Young's modulus [Pa]
    nu = 0.25  # Poisson's ratio
    coh = 1e6  # Cohesion [Pa] (example value)
    phi_d = 30  # Friction angle [deg]
    psi_d = 15  # Dilation angle [deg]

    # ------------------ Initial State ------------------
    NTENS = 6  # 3D stress state
    NDI = 3
    NSHR = 3
    NSTATV = 1

    # Initial strain (6-component strain, assumed all zero)
    strain_tensor = np.zeros(NTENS, dtype=np.float64)
    # Initial stress: assume all principal directions are at confining stress
    stress_tensor = np.array([confining_stress, confining_stress, confining_stress, 0, 0, 0], dtype=np.float64)

    # Other variables required by UMAT
    STATEV = np.zeros(NSTATV, dtype=np.float64)
    DDSDDE = np.zeros((NTENS, NTENS), dtype=np.float64)
    SSE = SPD = SCD = RPL = 0.0
    DDSDDT = np.zeros(NTENS, dtype=np.float64)
    # Time related (no thermal coupling here)
    TIME = np.array([0.0, 1.0], dtype=np.float64)
    DTIME = 1.0
    TEMP = DTEMP = 0.0
    PREDEF = DPRED = 0.0
    CMNAME = "MohrCoulomb"

    # Material properties array PROPS: [E, nu, coh, phi_d, psi_d]
    PROPS = np.array([E, nu, coh, phi_d, psi_d], dtype=np.float64)
    NPROPS = PROPS.size

    # Other auxiliary variables
    COORDS = np.zeros(3, dtype=np.float64)
    DROT = np.eye(3, dtype=np.float64)
    PNEWDT = 0.0
    CELENT = 0.0
    DFGRD0 = np.eye(3, dtype=np.float64)
    DFGRD1 = np.eye(3, dtype=np.float64)
    NOEL = NPT = LAYER = KSPT = KSTEP = KINC = 1

    # ----------------- Data Recording ------------------
    axial_strain_list = []  # Record axial (z) strain (index 2)
    lateral_strain_list = []  # Record lateral (x) strain (reverse sign)
    dev_stress_list = []  # Record deviatoric stress = (σ₁ - σ₃)/2
    principal_stress_list = []  # Optional: record principal stresses

    # ----------------- Simulation Process ------------------
    for i in range(nsteps):
        # Construct strain increment:
        # For axial strain: d_epsilon_z is applied as -d_epsilon_z (compression is negative),
        # and lateral strain (x-direction) is determined by Poisson's ratio.
        d_epsilon_tensor = np.array([nu * d_epsilon_z,
                                     nu * d_epsilon_z,
                                     -d_epsilon_z,
                                     0, 0, 0], dtype=np.float64)
        # Copy strain increment to DSTRAN (increment variable for UMAT)
        DSTRAN = d_epsilon_tensor.copy()

        # Update total strain
        strain_tensor = strain_tensor + d_epsilon_tensor

        # Call UMAT to update stress and state variables (UMAT updates stress_tensor and STATEV in-place)
        UMAT(stress_tensor, STATEV, DDSDDE, SSE, SPD, SCD,
             RPL, DDSDDT, None, 0.0, strain_tensor, DSTRAN,
             TIME, DTIME, TEMP, DTEMP, PREDEF, DPRED, CMNAME,
             NDI, NSHR, NTENS, NSTATV, PROPS, NPROPS, COORDS, DROT,
             PNEWDT, CELENT, DFGRD0, DFGRD1, NOEL, NPT, LAYER, KSPT, KSTEP, KINC)

        # Extract principal stresses using PrinStressAna
        principal = PrinStressAna(stress_tensor, NTENS)
        # Compute deviatoric stress: (σ₁ - σ₃)/2
        dev_stress = (principal[0] - principal[2]) / 2

        # Axial strain: use z-direction (index 2); take negative so that compression appears as positive on the plot
        axial_strain = -strain_tensor[2]
        # Lateral strain: use x-direction (index 0), reverse sign so that its sign is opposite to axial strain
        lateral_strain = -strain_tensor[0]

        axial_strain_list.append(axial_strain)
        lateral_strain_list.append(lateral_strain)
        dev_stress_list.append(dev_stress)
        principal_stress_list.append(principal)

    import matplotlib.pyplot as plt

    # ================== Plotting Results ======================
    plt.figure(figsize=(8, 6))

    # Plot axial strain vs deviatoric stress with smaller markers and thinner lines
    plt.plot(axial_strain_list, dev_stress_list, 'b-o', label='Axial Strain',
             markersize=0, linewidth=2)

    # Plot lateral strain vs deviatoric stress with smaller markers and thinner lines
    plt.plot(lateral_strain_list, dev_stress_list, 'r-s', label='Lateral Strain',
             markersize=0, linewidth=2)

    # Font settings for labels, title, ticks, and legend
    font_settings = {'family': 'serif', 'weight': 'normal', 'size': 14}  # Modify size as needed

    # Set axis labels with Times New Roman font
    plt.xlabel('Strain', fontdict=font_settings)
    plt.ylabel('Deviatoric Stress', fontdict=font_settings)

    # Set title with Times New Roman font
    plt.title('Mohr-Coulomb Model: Strain vs Deviatoric Stress',
              fontdict={'family': 'serif', 'weight': 'normal', 'size': 14})

    # Set legend with Times New Roman font using prop for font properties
    legend = plt.legend(prop={'family': 'serif', 'weight': 'normal', 'size': 13})

    # Make the legend draggable
    legend.set_draggable(True)

    # Set grid and apply the Times New Roman font for grid labels (axis ticks)
    plt.grid(True)
    plt.xticks(fontsize=12, family='serif')  # Set font and size for x-axis ticks
    plt.yticks(fontsize=12, family='serif')  # Set font and size for y-axis ticks

    # Invert the x-axis so that from zero to the right are negative values,
    # and from zero to the left are positive.
    ax = plt.gca()
    ax.invert_xaxis()

    # Set y-axis tick format in scientific notation (displaying numbers with a multiplier, e.g. 1e6)
    ax.ticklabel_format(style='sci', axis='y', scilimits=(6, 6))
    # Set the y-axis limits to make sure no negative values are shown
    plt.ylim(bottom=0)  # Only show y-axis values greater than or equal to 0
    # Show the plot
    plt.show()
