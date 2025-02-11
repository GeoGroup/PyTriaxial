# $$CC391
# -*- coding: UTF-8 -*-
"""
@Project ：PyTriaxial
@File    ：Duncan-Chang.py
@IDE     ：PyCharm 
@Author  ：Arjie.
@Date    ：2025-02-07 13:24 
"""
import numpy as np
import math
import sys


# ------------------------------------------------------------------------------
# Helper function: Sine of an angle (input angle in degrees)
def sind(angle_deg):
    """
    Calculate the sine value of an angle (in degrees)
    """
    return math.sin(math.radians(angle_deg))


# ------------------------------------------------------------------------------
# Helper function: Cotangent of an angle (input angle in degrees)
def cotand(angle_deg):
    """
    Calculate the cotangent value of an angle (in degrees)
    Note: A division by zero error will occur if the angle is close to 0° or 180°,
    please ensure the input is reasonable.
    """
    tan_val = math.tan(math.radians(angle_deg))
    if abs(tan_val) < 1e-12:
        return float('inf')
    return 1.0 / tan_val


# ------------------------------------------------------------------------------
# Helper function: Calculate the Euclidean norm of a vector
def norm2(vector):
    """
    Calculate the Euclidean norm of a vector
    """
    return np.linalg.norm(vector)


# ------------------------------------------------------------------------------
def Invariants(Sigma, nsigma):
    """
    Calculate the deviatoric stress vector of the stress vector Sigma, along with several stress invariants.

    Input:
      Sigma  -- numpy array, contains stress components, the length is determined by nsigma:
                nsigma = 3: Principal stresses, Sigma = [SigP_1, SigP_2, SigP_3]
                nsigma = 4: Plane stress or plane strain/axisymmetric problem,
                           Sigma = [sig_x, sig_y, sig_z, tau_xy] (sig_z is the out-of-plane stress)
                nsigma = 6: 3D stress state,
                           Sigma = [sig_x, sig_y, sig_z, tau_xy, tau_xz, tau_yz]
      nsigma -- Integer, the number of stress components (3, 4, or 6)

    Output:
      S       -- Deviatoric stress vector (numpy array), same size as Sigma,
                 represents components after removing the hydrostatic pressure.
      I1      -- First stress invariant, I1 = sigma_1 + sigma_2 + sigma_3
      J2      -- Second deviatoric stress invariant, J2 = 0.5 * sum(S^2)
      J3      -- Third deviatoric stress invariant (extended formula for 3D)
      lode    -- Lode angle (in radians), range is [-pi/6, pi/6]
      sin3lode-- Sine of three times the Lode angle, formula sin(3*lode) = - (3√3 * J3)/(2*J2^(3/2))
    """
    # Calculate I1 (sum of the first 3 components)
    I1 = np.sum(Sigma[:3])

    if nsigma == 3:
        # Sigma is the principal stress vector
        S = Sigma - I1 / 3.0  # Deviatoric stress vector
        J2 = 0.5 * np.sum(S ** 2)
        J3 = S[0] * S[1] * S[2]

    elif nsigma == 4:
        # Plane stress or axisymmetric case
        S = np.zeros(4, dtype=float)
        S[:3] = Sigma[:3] - I1 / 3.0
        S[3] = Sigma[3]
        J2 = 0.5 * np.sum(S[:3] ** 2) + S[3] ** 2
        J3 = (S[0] ** 3 + S[1] ** 3 + S[2] ** 3 + 3.0 * S[3] ** 2 * (S[0] + S[1])) / 3.0

    elif nsigma == 6:
        # 3D stress state
        S = np.zeros(6, dtype=float)
        S[:3] = Sigma[:3] - I1 / 3.0
        S[3:6] = Sigma[3:6]
        J2 = 0.5 * np.sum(S[:3] ** 2) + np.sum(S[3:6] ** 2)
        J3 = (np.sum(S[:3] ** 3) + 6.0 * S[3] * S[4] * S[5] +
              3.0 * (S[0] * (S[3] ** 2 + S[4] ** 2) +
                     S[1] * (S[3] ** 2 + S[5] ** 2) +
                     S[2] * (S[4] ** 2 + S[5] ** 2))) / 3.0
    else:
        raise ValueError("nsigma must be 3, 4, or 6.")

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


# ------------------------------------------------------------------------------
# PrinStressAna function: Calculate principal stresses and sort in descending order
def PrinStressAna(Sigma):
    """
    Calculate the principal stresses of the stress vector Sigma using analytical expressions and sort them in descending order.

    For plane problems (nsigma = 4):
      Input Sigma is a length 4 array, ordered as:
         [sig_x, sig_y, sig_z, tau_xy]  (sig_z is the out-of-plane stress)
      Calculation method:
         sig_av = 0.5*(sig_x + sig_y)
         sig_hj = sqrt((0.5*(sig_x - sig_y))^2 + tau_xy^2)
         Principal stresses are then:
            SigP[0] = sig_av + sig_hj
            SigP[1] = sig_av - sig_hj
            SigP[2] = sig_z
         Then reorder to ensure SigP[0] ≥ SigP[1] ≥ SigP[2].

    For 3D stress state (nsigma = 6):
      Input Sigma is a length 6 array, ordered as:
         [sig_x, sig_y, sig_z, tau_xy, tau_xz, tau_yz]
      Call Invariants to calculate invariants and Lode angle, using the following formulas:
         sigm = I1/3.0   (hydrostatic pressure)
         If sqrt(J2) < 1.0e-12 * abs(I1), then directly take:
            SigP = [sig_x, sig_y, sig_z]
         Otherwise:
            sqJ2 = 1.154700538379252 * sqrt(J2)   (2/sqrt(3))
            SigP[0] = sqJ2 * sin(lode + 2.094395102393195) + sigm
            SigP[1] = sqJ2 * sin(lode) + sigm
            SigP[2] = sqJ2 * sin(lode - 2.094395102393195) + sigm

    For nsigma = 3 (already principal stresses), directly return Sigma sorted in descending order.

    Parameters:
      Sigma: numpy array, stress components. Length is 4 for plane problems, 6 for 3D problems, or 3 for principal stresses.

    Returns:
      SigP: numpy array of length 3, principal stresses [σ₁, σ₂, σ₃] sorted in descending order.
    """
    nsigma = len(Sigma)
    SigP = np.zeros(3, dtype=float)

    if nsigma == 3:
        # Already principal stresses, sort in descending order
        SigP = np.sort(Sigma)[::-1]

    elif nsigma == 4:
        # Plane stress or axisymmetric problem
        sig_av = 0.5 * (Sigma[0] + Sigma[1])
        sig_hj = math.sqrt((0.5 * (Sigma[0] - Sigma[1])) ** 2 + Sigma[3] ** 2)
        SigP[0] = sig_av + sig_hj
        SigP[1] = sig_av - sig_hj
        SigP[2] = Sigma[2]  # Out-of-plane stress

        # Sort
        if Sigma[2] > SigP[0]:
            SigP = np.array([Sigma[2], SigP[0], SigP[1]])
        elif Sigma[2] > SigP[1]:
            SigP = np.array([SigP[0], Sigma[2], SigP[1]])
        # If the above conditions aren't satisfied, SigP[0] ≥ SigP[1] and SigP[2] is the smallest.

    elif nsigma == 6:
        # 3D stress state
        # Call Invariants function (returns values in order: S, I1, J2, J3, lode, sin3lode)
        _, I1, J2, J3, lode, _ = Invariants(Sigma, nsigma)
        sigm = I1 / 3.0  # Hydrostatic pressure

        if math.sqrt(J2) < 1.0e-12 * abs(I1):
            SigP[0] = Sigma[0]
            SigP[1] = Sigma[1]
            SigP[2] = Sigma[2]
        else:
            sqJ2 = 1.154700538379252 * math.sqrt(J2)  # 2/sqrt(3)
            SigP[0] = sqJ2 * math.sin(lode + 2.094395102393195) + sigm
            SigP[1] = sqJ2 * math.sin(lode) + sigm
            SigP[2] = sqJ2 * math.sin(lode - 2.094395102393195) + sigm
    else:
        raise ValueError("Sigma length must be 3 (principal stresses), 4 (plane stress), or 6 (3D stress).")

    # To ensure strict descending order, sort again (in descending order)
    SigP = np.sort(SigP)[::-1]
    return SigP


# ------------------------------------------------------------------------------
# Calculate the elastic stiffness matrix (6x6 matrix)
def GetDDSDDE(E, nu):
    """
    Calculate the elastic stiffness matrix ddsdde (6x6) based on the current Young's modulus E and Poisson's ratio nu

    Parameters:
      E  -- Young's modulus (float)
      nu -- Poisson's ratio (float)

    Returns:
      ddsdde -- 6x6 numpy array, representing the elastic stiffness matrix
    """
    ddsdde = np.zeros((6, 6), dtype=float)

    # Assign values for the first three stress components
    ddsdde[0, 0] = 1.0 - nu
    ddsdde[0, 1] = nu
    ddsdde[0, 2] = nu

    ddsdde[1, 0] = nu
    ddsdde[1, 1] = 1.0 - nu
    ddsdde[1, 2] = nu

    ddsdde[2, 0] = nu
    ddsdde[2, 1] = nu
    ddsdde[2, 2] = 1.0 - nu

    # Shear components
    shear_term = 0.5 * (1.0 - 2.0 * nu)
    ddsdde[3, 3] = shear_term
    ddsdde[4, 4] = shear_term
    ddsdde[5, 5] = shear_term

    ddsdde *= E / ((1.0 + nu) * (1.0 - 2.0 * nu))

    return ddsdde


# ------------------------------------------------------------------------------
# Function to update material modulus, calling PrinStressAna to calculate principal stresses
def GetModulus(stress, props, s_history):
    """
    Calculate the Young's modulus E and parameter s based on the current stress state and material parameters.

    Parameters:
      stress    -- Length 6 numpy array representing the stress tensor
      props     -- Material constants numpy array, ordered as follows:
                   props[0]: E50
                   props[1]: Eur
                   props[2]: nu  (Poisson's ratio)
                   props[3]: phi (friction angle in degrees)
                   props[4]: c   (cohesion)
                   props[5]: Rf
      s_history -- Historical parameter s_history (float)

    Returns:
      E -- Calculated Young's modulus
      s -- Current parameter s
    """
    E50 = props[0]
    Eur = props[1]
    phi = props[3]
    c = props[4]
    Rf = props[5]

    Ei = 2.0 * E50 / (2.0 - Rf)

    # Call PrinStressAna to calculate principal stresses, return sorted principal stresses
    stress_prin = PrinStressAna(stress)
    q = stress_prin[0] - stress_prin[2]

    if stress_prin[0] >= c * cotand(phi):
        qf = 0.0
        s = 1.0
        print("warning: principal stress exceeds limit, shear resistance lost", file=sys.stderr)
    else:
        qf = 2.0 * sind(phi) / (1.0 - sind(phi)) * (-stress_prin[0] + c * cotand(phi))
        s = q / qf

    if s >= 1.0:
        E = 0.0
    elif s >= s_history:
        E = Ei * (1.0 - Rf * s) ** 2
    else:
        E = Eur

    return E, s


# ------------------------------------------------------------------------------
# DuncanChang Subroutine: Main calculation subroutine to update material state
def DuncanChang(stress, statev, ddsdde, stran, dstran,
                ndi, nshr, ntens, nstatv, props, nprops, drot):
    """
    Main calculation subroutine for the DuncanChang constitutive model, used to update stress, state variables, and stiffness matrix.

    Parameters:
      stress  -- Stress tensor, numpy array, length ntens (in/out)
      statev  -- State variable array, numpy array, length nstatv (in/out)
                 Where:
                   statev[0] stores the historical parameter s_history
                   statev[1] records the iteration count k1
                   statev[2] records the final relative error RE
      ddsdde  -- Stiffness matrix, numpy array, shape (ntens, ntens) (in/out)
      stran   -- Total strain (initial state, strain value array)
      dstran  -- Strain increment (array)
      ndi, nshr, ntens, nstatv, nprops -- Integer parameters representing the number of normal stress components, shear components, stress tensor size,
                                          state variable count, and number of material constants
      props   -- Material constants array, numpy array, length nprops
                Where: props[0]: E50, props[1]: Eur, props[2]: nu, props[3]: phi (degrees),
                      props[4]: c, props[5]: Rf
      drot    -- Rotation matrix (3x3), numpy array (in) — not used in this example

    Returns:
      Updated stress, statev, ddsdde
    """
    toler = 0.0001
    nu = props[2]

    T = 1.0
    dt = T
    T_remain = T

    s_history = statev[0]
    if s_history < 0.001:
        s_history = 0.001

    RE = 0.0
    k1_final = 0

    for k1 in range(1, 20001):  # Max 20,000 iterations
        k1_final = k1

        # Step strain increment
        dstran_step = dt * dstran

        # Step 1: Calculate E and s from the current stress and then predict stress stress1
        E, s = GetModulus(stress, props, s_history)
        De1 = GetDDSDDE(E, nu)
        stress1 = stress + np.dot(De1, dstran_step)

        # Step 2: Calculate E and s from the predicted stress stress1 and then predict stress stress2
        E, s = GetModulus(stress1, props, s_history)
        De2 = GetDDSDDE(E, nu)
        stress2 = stress + np.dot(De2, dstran_step)

        # Calculate relative error RE
        numerator = norm2(stress2 - stress1)
        denominator = norm2(stress2 + stress1)
        RE = numerator / denominator if denominator != 0.0 else 0.0

        if RE > toler:
            # If error is large, shorten step size
            dt = 0.8 * math.sqrt(toler / RE) * dt
        else:
            # Accept the result for this step: update stress as the average of the two predictions
            stress = 0.5 * (stress1 + stress2)

            # Update historical stress ratio
            E, s = GetModulus(stress, props, s_history)
            if s > s_history:
                s_history = s

            T_remain -= dt
            if T_remain <= dt:
                dt = T_remain
            if abs(T_remain) < 1e-12:
                break

    # After the loop ends, calculate the final stiffness matrix using the updated stress
    E, s = GetModulus(stress, props, s_history)
    ddsdde[:, :] = GetDDSDDE(E, nu)

    # Update state variables
    if s > statev[0]:
        statev[0] = s
    statev[1] = k1_final
    statev[2] = RE

    return stress, statev, ddsdde


# ------------------------------------------------------------------------------
if __name__ == "__main__":
    import numpy as np
    import math
    import matplotlib.pyplot as plt

    # ================== Simulation Parameters ======================
    # Material parameters
    E50 = 10000.0  # E50
    Eur = 5000.0  # Eur
    nu = 0.4  # Poisson's ratio
    phi = 20.0  # Friction angle [degrees]
    c = 30.0  # Cohesion
    rf = 0.9  # Parameter rf

    PROPS = np.array([E50, Eur, nu, phi, c, rf], dtype=np.float64)
    NPROPS = PROPS.size

    # Initial confining stress: -50 MPa (compression is negative)
    confining_stress = -50.0  # in MPa

    # Loading stages:
    # Stage 1: Load axial strain from 0 to -0.05 (compression)
    steps_stage1 = 50
    # Stage 2: Unload axial strain from -0.05 to -0.04 (unloading 0.01)
    steps_stage2 = 10
    # Stage 3: Further load axial strain from -0.04 to -0.2 (additional -0.16)
    steps_stage3 = 50

    stage1_increments = np.full(steps_stage1, -0.05 / steps_stage1)  # each step -0.001
    stage2_increments = np.full(steps_stage2, 0.01 / steps_stage2)  # each step +0.001 (unloading)
    stage3_increments = np.full(steps_stage3, -0.16 / steps_stage3)  # each step -0.0032

    all_increments = np.concatenate([stage1_increments, stage2_increments, stage3_increments])

    # ================== Initial State ======================
    NTENS = 6  # 3D stress state (6 components)
    NDI = 3  # Number of normal stress components
    NSHR = 3  # Number of shear stress components
    NSTATV = 3  # Number of state variables

    # Initial strain tensor (6 components, all zero)
    strain_tensor = np.zeros(NTENS, dtype=np.float64)

    # Initial stress tensor: normal stresses all equal to confining stress (compression) and zero shear stresses
    stress_tensor = np.array([confining_stress, confining_stress, confining_stress, 0.0, 0.0, 0.0], dtype=np.float64)

    # Initialize state variables and stiffness matrix (to be updated in the UMAT)
    STATEV = np.zeros(NSTATV, dtype=np.float64)
    DDSDDE = np.zeros((NTENS, NTENS), dtype=np.float64)

    # Rotation matrix (not used in this example, so pass identity matrix)
    DROT = np.eye(3, dtype=np.float64)

    # ================== Data Recording ======================
    axial_strain_list = []  # Record axial strain (z-direction, negative for compression)
    lateral_strain_list = []  # Record lateral strain (average of x, y strains, positive for expansion)
    deviatoric_stress_list = []  # Record deviatoric stress: (σ₁ - σ₃) / 2
    principal_stress_list = []  # Record principal stresses (3 components)


    # ----------------- Helper function: Construct strain increment -----------------
    def construct_strain_increment(d_epsilon_z):
        """
        Construct a 6-component strain increment vector from the given axial strain increment d_epsilon_z.
        The axial strain (z-direction, index 2) is taken as d_epsilon_z.
        Lateral strains (x and y, indices 0 and 1) are assumed as -nu * d_epsilon_z,
        ensuring that when axial strain is compressive (negative), lateral strain is tensile (positive).
        """
        d_strain = np.zeros(NTENS, dtype=np.float64)
        d_strain[2] = d_epsilon_z
        d_strain[0] = -nu * d_epsilon_z
        d_strain[1] = -nu * d_epsilon_z
        return d_strain


    # ================== Loading Process Simulation ======================
    for d_epsilon_z in all_increments:
        # Construct strain increment (including lateral components)
        d_strain = construct_strain_increment(d_epsilon_z)

        # Save the current total strain before updating (for UMAT input)
        stran = strain_tensor.copy()
        # Update total strain tensor
        strain_tensor += d_strain

        # Call the constitutive model update function (DuncanChang)
        # DuncanChang updates stress_tensor, STATEV, DDSDDE in-place.
        stress_tensor, STATEV, DDSDDE = DuncanChang(
            stress_tensor, STATEV, DDSDDE, stran, d_strain,
            NDI, NSHR, NTENS, NSTATV, PROPS, NPROPS, DROT
        )

        # Extract principal stresses (for a 3D stress state, stress_tensor has 6 components)
        principal = PrinStressAna(stress_tensor)
        # Compute deviatoric stress: (σ₁ - σ₃) / 2
        dev_stress = (principal[0] - principal[2]) / 2.0

        # Record axial strain (z-direction, negative value for compression)
        axial_strain = strain_tensor[2]
        # Record lateral strain: average of x and y strains (assumed positive for expansion)
        lateral_strain = (strain_tensor[0] + strain_tensor[1]) / 2.0

        axial_strain_list.append(axial_strain)
        lateral_strain_list.append(lateral_strain)
        deviatoric_stress_list.append(dev_stress)
        principal_stress_list.append(principal.copy())

        axial_strain_list.insert(0, 0)  # Add 0 to axial strain list
        deviatoric_stress_list.insert(0, 0)  # Add 0 to deviatoric stress list
        lateral_strain_list.insert(0, 0)  # Add 0 to lateral strain list

    # ================== Plotting ======================

    plt.figure(figsize=(8, 6))

    # Plot axial strain vs. deviatoric stress (axial strain is negative)
    plt.plot(axial_strain_list, deviatoric_stress_list, 'b-o', label='Axial Strain',
             markersize=0, linewidth=2)

    # Plot lateral strain vs. deviatoric stress (lateral strain is positive)
    plt.plot(lateral_strain_list, deviatoric_stress_list, 'r-s', label='Lateral Strain',
             markersize=0, linewidth=2)

    # Font settings for labels, title, ticks, and legend
    font_settings = {'family': 'serif', 'weight': 'normal', 'size': 14}  # Modify size as needed

    # Set axis labels with Times New Roman font
    plt.xlabel('Strain', fontdict=font_settings)
    plt.ylabel('Deviatoric Stress', fontdict=font_settings)

    # Set title with Times New Roman font
    plt.title('Duncan-Chang Model: Strain vs Deviatoric Stress',
              fontdict={'family': 'serif', 'weight': 'normal', 'size': 14})

    # Set legend with Times New Roman font using prop for font properties
    legend = plt.legend(prop={'family': 'serif', 'weight': 'normal', 'size': 13})
    # Make the legend draggable
    legend.set_draggable(True)

    # Set grid and apply the Times New Roman font for grid labels (axis ticks)
    plt.grid(True)
    plt.xticks(fontsize=12, family='serif')  # Set font and size for x-axis ticks
    plt.yticks(fontsize=12, family='serif')  # Set font and size for y-axis ticks

    # Set x-axis limits: left end from lateral strain maximum (positive) and right end from axial strain minimum (negative)
    if lateral_strain_list and axial_strain_list:
        x_left = max(lateral_strain_list)
        x_right = min(axial_strain_list)
        plt.xlim(x_left, x_right)

    # Set the y-axis limits to make sure no negative values are shown
    plt.ylim(bottom=0)  # Only show y-axis values greater than or equal to 0

    # Show the plot
    plt.show()
