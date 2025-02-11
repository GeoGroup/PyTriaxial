# $$CC391
# -*- coding: UTF-8 -*-
"""
@Project ：PyTriaxial
@File    ：Elastic.py
@IDE     ：PyCharm 
@Author  ：Arjie.
@Date    ：2025-02-07 16:32 
"""
import numpy as np
import matplotlib.pyplot as plt
import math


def UMAT(STRESS, STATEV, DDSDDE, SSE, SPD, SCD,
         RPL, DDSDDT, DRPLDE, DRPLDT,
         STRAN, DSTRAN, TIME, DTIME, TEMP, DTEMP, PREDEF, DPRED, CMNAME,
         NDI, NSHR, NTENS, NSTATV, PROPS, NPROPS, COORDS, DROT, PNEWDT,
         CELENT, DFGRD0, DFGRD1, NOEL, NPT, LAYER, KSPT, JSTEP, KINC):
    """
    Isotropic linear elastic constitutive model

    Calculates stress increment based on strain increment DSTRAN and updates stress STRESS,
    while constructing the stiffness matrix DDSDDE.
    Material properties are obtained from PROPS, where PROPS[0] is the Young's modulus EE,
    and PROPS[1] is the Poisson's ratio ENU.
    Note: This program only considers 3D stress state (6 components: the first 3 are normal stress/strain,
    and the last 3 are shear components).
    """
    # Define constants
    ONE = 1.0
    TWO = 2.0

    # -------------------------------
    # Material properties
    EE = PROPS[0]  # Young's modulus
    ENU = PROPS[1]  # Poisson's ratio

    # Calculate shear modulus EG and Lamé's first parameter ELAM
    EG = EE / (TWO * (ONE + ENU))
    ELAM = (ENU * EE) / ((ONE + ENU) * (ONE - TWO * ENU))

    # -------------------------------
    # Construct stiffness matrix DDSDDE
    # Top-left part: normal stress/strain components (NDI components), assign ELAM and add 2*EG to the diagonal
    for i in range(NDI):
        for j in range(NDI):
            DDSDDE[i, j] = ELAM
        DDSDDE[i, i] = TWO * EG + ELAM
    # Bottom-right part: shear components, assign EG
    for i in range(NDI, NTENS):
        DDSDDE[i, i] = EG

    # -------------------------------
    # Calculate stress increment and update stress STRESS (in-place update)
    DSTRESS = np.zeros(NTENS)
    for i in range(NTENS):
        for j in range(NTENS):
            DSTRESS[i] = DDSDDE[i, j] * DSTRAN[j]
            STRESS[i] += DSTRESS[i]

    return


def Invariants(Sigma):
    """
    --------------------------------------------------------------------------
    Calculates the deviatoric stress vector S and stress invariants for the 3D stress state Sigma.

    Input:
      Sigma: 1D numpy array of length 6, corresponding to the 3D stress state,
             in the order [σ_x, σ_y, σ_z, τ_xy, τ_xz, τ_yz].

    Output:
      S       : Deviatoric stress vector, the stress components without the hydrostatic part (length 6).
      I1      : First stress invariant, I1 = σ_x + σ_y + σ_z.
      J2      : Second deviatoric stress invariant, J2 = 0.5*(S_x²+S_y²+S_z²) + (τ_xy²+τ_xz²+τ_yz²).
      J3      : Third deviatoric stress invariant, calculated according to the code formula in the paper.
      lode    : Lode angle (in radians), satisfies -π/6 ≤ lode ≤ π/6.
      sin3lode: Sine of three times the Lode angle.

    Note:
      This function directly refers to the Fortran code you provided and only considers nsigma == 6 for 3D stress states.
    --------------------------------------------------------------------------
    """
    nsigma = len(Sigma)
    if nsigma != 6:
        raise ValueError("The Invariants function only supports 3D stress states (nsigma=6)")

    # First stress invariant: I1 = σ_x + σ_y + σ_z
    I1 = Sigma[0] + Sigma[1] + Sigma[2]

    # Calculate deviatoric stress vector S:
    # For the first three components: S_i = σ_i - I1/3; the last 3 components (shear) remain unchanged
    S = np.empty(6)
    S[0:3] = Sigma[0:3] - I1 / 3.0
    S[3:6] = Sigma[3:6]

    # Second deviatoric stress invariant J2
    J2 = 0.5 * np.sum(S[0:3] ** 2) + np.sum(S[3:6] ** 2)

    # Third deviatoric stress invariant J3 (according to the Fortran code formula)
    J3 = (np.sum(S[0:3] ** 3) +
          6.0 * S[3] * S[4] * S[5] +
          3.0 * (S[0] * (S[3] ** 2 + S[4] ** 2) +
                 S[1] * (S[3] ** 2 + S[5] ** 2) +
                 S[2] * (S[4] ** 2 + S[5] ** 2))
          ) / 3.0

    # Calculate sin3lode: it is 0 when J2 ≤ 0
    if J2 > 0.0:
        sin3lode = -2.598076211353316 * J3 / (J2 ** 1.5)
    else:
        sin3lode = 0.0

    # Calculate Lode angle lode based on sin3lode, ensuring it is within [-pi/6, pi/6]
    if sin3lode <= -1.0:
        lode = -0.5235987755982988  # -pi/6
    elif sin3lode >= 1.0:
        lode = 0.5235987755982988  # +pi/6
    else:
        lode = math.asin(sin3lode) / 3.0

    return S, I1, J2, J3, lode, sin3lode


def PrinStressAna(Sigma):
    """
    --------------------------------------------------------------------------
    Calculates the principal stresses for the 3D stress state Sigma using an analytical method,
    and orders them in descending order.

    Input:
      Sigma: 1D numpy array of length 6, in the order [σ_x, σ_y, σ_z, τ_xy, τ_xz, τ_yz].

    Output:
      SigP: 1D numpy array of length 3, principal stresses [σ₁, σ₂, σ₃] in descending order.

    Note:
      This function calls the Invariants function to compute the stress invariants,
      and only considers nsigma == 6 for 3D stress states.
    --------------------------------------------------------------------------
    """
    nsigma = len(Sigma)
    if nsigma != 6:
        raise ValueError("The PrinStressAna function only supports 3D stress states (nsigma=6)")

    # Call Invariants to compute deviatoric stress and invariants
    S, I1, J2, J3, lode, sin3lode = Invariants(Sigma)

    # Hydrostatic pressure (mean stress)
    sigm = I1 / 3.0

    # When sqrt(J2) is sufficiently small, assume deviatoric stress is zero, and principal stresses are just the original normal stresses
    if np.sqrt(J2) < 1.0e-12 * abs(I1):
        SigP1, SigP2, SigP3 = Sigma[0], Sigma[1], Sigma[2]
    else:
        # sqJ2 = 2/sqrt(3)*sqrt(J2)  (2/sqrt(3) ≈ 1.154700538379252)
        sqJ2 = 1.154700538379252 * np.sqrt(J2)
        # Calculate principal stresses based on analytical expressions:
        # Constant 2.094395102393195 is 2*pi/3
        SigP1 = sqJ2 * math.sin(lode + 2.094395102393195) + sigm
        SigP2 = sqJ2 * math.sin(lode) + sigm
        SigP3 = sqJ2 * math.sin(lode - 2.094395102393195) + sigm

    # Ensure principal stresses are ordered in descending order
    SigP = np.array([SigP1, SigP2, SigP3])
    SigP = -np.sort(-SigP)  # Sort in descending order
    return SigP


# ----------------------------------------------------------------------
# Example for isotropic linear elastic triaxial compression test
if __name__ == "__main__":
    # ================== Example parameters ======================
    # Rock triaxial compression test parameters (convention: tension is positive, compression is negative)
    confining_stress = -1e6  # Confining stress [Pa] (compression is negative)
    total_axial_strain = -0.035  # Total axial strain (compression is negative)
    nsteps = 1000  # Number of test steps
    d_epsilon_z = total_axial_strain / nsteps  # Increment of axial strain per step

    # ---------------- Material properties ----------------
    E = 1e8  # Young's modulus [Pa]
    nu = 0.25  # Poisson's ratio
    PROPS = np.array([E, nu], dtype=np.float64)
    NPROPS = PROPS.size

    # ---------------- Initial state ----------------
    NTENS = 6  # 6 components for 3D stress state (3 normal stresses + 3 shear stresses)
    NDI = 3  # Number of normal stress/strain components (x, y, z)
    NSHR = 3  # Number of shear components
    NSTATV = 1  # Number of state variables

    # Initial total strain (all zeros, 6 components)
    strain_tensor = np.zeros(NTENS, dtype=np.float64)
    # Initial stress: confining stress applied in all principal directions (compression stress is confining_stress), shear stress is 0
    stress_tensor = np.array([confining_stress, confining_stress, confining_stress, 0, 0, 0], dtype=np.float64)

    # ---------------- Other variables required by UMAT ----------------
    STATEV = np.zeros(NSTATV, dtype=np.float64)
    DDSDDE = np.zeros((NTENS, NTENS), dtype=np.float64)
    SSE = SPD = SCD = RPL = 0.0
    DDSDDT = np.zeros(NTENS, dtype=np.float64)
    TIME = np.array([0.0, 1.0], dtype=np.float64)
    DTIME = 1.0
    TEMP = DTEMP = 0.0
    PREDEF = DPRED = 0.0
    CMNAME = "Elastic3D"
    COORDS = np.zeros(3, dtype=np.float64)
    DROT = np.eye(3, dtype=np.float64)
    PNEWDT = 0.0
    CELENT = 0.0
    DFGRD0 = np.eye(3, dtype=np.float64)
    DFGRD1 = np.eye(3, dtype=np.float64)
    NOEL = NPT = LAYER = KSPT = KSTEP = KINC = 1

    # ---------------- Record test process data ----------------
    axial_strain_list = []  # Record axial strain in the z direction (compression should be negative)
    lateral_strain_list = []  # Record lateral strain in the x direction (expansion should be positive)
    dev_stress_list = []  # Record deviatoric stress ((σ₁ - σ₃)/2)
    principal_stress_list = []  # Record principal stress vectors

    # ---------------- Simulate test process ----------------
    for i in range(nsteps):
        # Construct strain increment:
        # Axial (z direction) strain takes d_epsilon_z (negative for compression);
        # Lateral (x, y direction) strain follows Poisson's ratio, taking nu*d_epsilon_z
        d_epsilon_tensor = np.array([
            nu * d_epsilon_z,  # x direction
            nu * d_epsilon_z,  # y direction
            -d_epsilon_z,  # z direction (since d_epsilon_z is negative, -d_epsilon_z is positive)
            0, 0, 0
        ], dtype=np.float64)

        # Update total strain
        strain_tensor += d_epsilon_tensor

        # Call UMAT to update stress (in-place update of stress_tensor)
        UMAT(stress_tensor, STATEV, DDSDDE, SSE, SPD, SCD,
             RPL, DDSDDT, None, 0.0, strain_tensor, d_epsilon_tensor,
             TIME, DTIME, TEMP, DTEMP, PREDEF, DPRED, CMNAME,
             NDI, NSHR, NTENS, NSTATV, PROPS, NPROPS, COORDS, DROT,
             PNEWDT, CELENT, DFGRD0, DFGRD1, NOEL, NPT, LAYER, KSPT, KSTEP, KINC)

        # Calculate principal stress for the current step (call the analytical method)
        principal = PrinStressAna(stress_tensor)
        # Calculate deviatoric stress: (σ₁ - σ₃)/2
        dev_stress = (principal[0] - principal[2]) / 2.0

        # For plotting:
        # - Axial strain (z direction) is negative for compression, shown on the right side (recorded as -strain_tensor[2])
        # - Lateral strain (x direction) is positive for expansion, shown on the left side (recorded as -strain_tensor[0])
        axial_strain = -strain_tensor[2]
        lateral_strain = -strain_tensor[0]

        axial_strain_list.append(axial_strain)
        lateral_strain_list.append(lateral_strain)
        dev_stress_list.append(dev_stress)
        principal_stress_list.append(principal.copy())

    # ================== Plotting ======================
    plt.figure(figsize=(8, 6))

    # Plot axial strain vs. deviatoric stress (axial strain is negative)
    plt.plot(axial_strain_list, dev_stress_list, 'b-', label='Axial Strain', linewidth=2)

    # Plot lateral strain vs. deviatoric stress (lateral strain is positive)
    plt.plot(lateral_strain_list, dev_stress_list, 'r-', label='Lateral Strain', linewidth=2)

    # Font settings for labels, title, ticks, and legend
    font_settings = {'family': 'serif', 'weight': 'normal', 'size': 14}  # Modify size as needed

    # Set axis labels with Times New Roman font
    plt.xlabel('Strain', fontdict=font_settings)
    plt.ylabel('Deviatoric Stress', fontdict=font_settings)

    # Set title with Times New Roman font
    plt.title('Linear Elastic Model: Strain vs Deviatoric Stress',
              fontdict={'family': 'serif', 'weight': 'normal', 'size': 14})

    # Set legend with Times New Roman font using prop for font properties
    legend = plt.legend(prop={'family': 'serif', 'weight': 'normal', 'size': 13})

    # Make the legend draggable
    legend.set_draggable(True)

    # Set grid and apply the Times New Roman font for grid labels (axis ticks)
    plt.grid(True)
    plt.xticks(fontsize=12, family='serif')  # Set font and size for x-axis ticks
    plt.yticks(fontsize=12, family='serif')  # Set font and size for y-axis ticks

    # Invert x-axis so that zero point is to the right (positive to left, negative to right)
    plt.gca().invert_xaxis()

    # Set the y-axis limits to make sure no negative values are shown
    plt.ylim(bottom=0)  # Only show y-axis values greater than or equal to 0
    # Show the plot
    plt.show()
