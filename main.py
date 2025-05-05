import numpy as np
from CoolProp.CoolProp import PropsSI
import math

# ==============================================================================
# Minimum temperature clamp to avoid calling CoolProp below melting point (for water ~273.15 K)
# ===============================================================================
MIN_TEMP_K = 274

# ==============================================================================
# Function to retrieve fluid properties (density, viscosity, specific heat, conductivity, Prandtl number)
# using CoolProp for given temperature T (K) and pressure P (Pa).
# Clamps T to MIN_TEMP_K to avoid errors below freezing/melting.
# ===============================================================================
def fluid_properties(T, P, fluid='Water'):
    # Clamp temperature to avoid unsupported region
    T_use = max(T, MIN_TEMP_K)
    rho = PropsSI('D', 'T', T_use, 'P', P, fluid)
    mu  = PropsSI('VISCOSITY', 'T', T_use, 'P', P, fluid)
    Cp  = PropsSI('C', 'T', T_use, 'P', P, fluid)
    k   = PropsSI('CONDUCTIVITY', 'T', T_use, 'P', P, fluid)
    Pr  = Cp * mu / k
    return rho, mu, Cp, k, Pr

# ==============================================================================
# Compute convective heat transfer coefficient and Reynolds number using Dittus-Boelter.
# Inputs:
#   rho     - fluid density (kg/m3)
#   v       - flow velocity (m/s)
#   De      - characteristic diameter (m)
#   mu      - dynamic viscosity (Pa·s)
#   Cp      - specific heat capacity (J/kg·K)
#   k       - thermal conductivity (W/m·K)
#   heating - True for heating correlation exponent, False for cooling.
# Returns:
#   h       - convective heat transfer coefficient (W/m2·K)
#   Re      - Reynolds number
# ===============================================================================
def heat_transfer_coeff(rho, v, De, mu, Cp, k, heating=True):
    Re = rho * v * De / mu
    Pr = Cp * mu / k
    n  = 0.4 if heating else 0.3  # exponent
    Nu = 0.023 * Re**0.8 * Pr**n
    h  = Nu * k / De
    return h, Re

# ==============================================================================
# Compute Darcy-Weisbach friction factor via Serghides' explicit approximation of Colebrook.
# Inputs:
#   Re  - Reynolds number
#   eps - pipe roughness (m)
#   De  - hydraulic diameter (m)
# ===============================================================================
def friction_factor(Re, eps, De):
    if Re < 2000:
        return 64.0 / Re  # laminar
    # Colebrook-Serghides explicit
    A = -2.0 * math.log10(eps/(3.7*De) + 12/Re)
    B = -2.0 * math.log10(eps/(3.7*De) + 2.51*A/Re)
    C = -2.0 * math.log10(eps/(3.7*De) + 2.51*B/Re)
    return (A - (B - A)**2/(C - 2*B + A))**-2

# ==============================================================================
# Prompt user for all required Plate Heat Exchanger (PHE) parameters.
# Returns tuple of all inputs in SI units.
# ===============================================================================
def interactive_inputs():
    print("\n--- Enter PHE Parameters ---")
    T_hot_in  = float(input("Hot inlet temp (°C): ")) + 273.15
    T_cold_in = float(input("Cold inlet temp (°C): ")) + 273.15
    P_hot_in  = float(input("Hot inlet pressure (Pa): "))
    P_cold_in = float(input("Cold inlet pressure (Pa): "))
    m_hot     = float(input("Hot mass flow rate (kg/s): "))
    m_cold    = float(input("Cold mass flow rate (kg/s): "))
    L         = float(input("Channel length (m): "))
    A         = float(input("Total heat transfer area (m²): "))
    De        = float(input("Equivalent hydraulic diameter (m): "))
    kw        = float(input("Plate conductivity (W/m·K): "))
    eps       = float(input("Surface roughness (m, e.g., 1e-5): "))
    segments  = int(input("Number of segments (e.g. 10): "))
    return T_hot_in, T_cold_in, P_hot_in, P_cold_in, m_hot, m_cold, L, A, De, kw, eps, segments

# ==============================================================================
# Iterative solver: marches through N segments, updating temperature, pressure, and wall temp,
# iterating heat transfer until convergence of Q.
# ===============================================================================
def iterative_solver():
    Th0, Tc0, Ph0, Pc0, mh, mc, L, A, De, kw, eps, N = interactive_inputs()
    dz    = L / N
    A_seg = A / N
    Th = np.zeros(N+1); Tc = np.zeros(N+1)
    Ph = np.zeros(N+1); Pc = np.zeros(N+1)
    Tw = np.zeros(N)
    Th[0], Tc[0], Ph[0], Pc[0] = Th0, Tc0, Ph0, Pc0
    Q_guess = 1000.0
    tol     = 1e-6
    for _ in range(1000):
        Q_total = 0.0
        for i in range(N):
            Th_mean = max((Th[i] + Th[i+1] if i+1<=N else Th[i])/2, MIN_TEMP_K)
            Tc_mean = max((Tc[i] + Tc[i+1] if i+1<=N else Tc[i])/2, MIN_TEMP_K)
            rho_h, mu_h, Cp_h, k_h, _ = fluid_properties(Th_mean, Ph[i])
            rho_c, mu_c, Cp_c, k_c, _ = fluid_properties(Tc_mean, Pc[i])
            v_h = mh/(rho_h*A_seg); v_c = mc/(rho_c*A_seg)
            hh, Re_h = heat_transfer_coeff(rho_h, v_h, De, mu_h, Cp_h, k_h)
            hc, Re_c = heat_transfer_coeff(rho_c, v_c, De, mu_c, Cp_c, k_c, heating=False)
            U = 1/(1/hh + De/kw + 1/hc)
            dT1 = Th[i] - (Tc[i+1] if i+1<=N else Tc[i])
            dT2 = (Th[i+1] if i+1<=N else Th[i]) - Tc[i]
            dTlm = (dT1 - dT2)/math.log(abs(dT1/dT2))
            Q = U * A_seg * dTlm
            Q_total += Q
            Th[i+1] = max(Th[i] - Q/(mh*Cp_h), MIN_TEMP_K)
            Tc[i+1] = max(Tc[i] + Q/(mc*Cp_c), MIN_TEMP_K)
            Tw[i]    = (hh*Th[i+1] + hc*Tc[i+1])/(hh+hc)
            f_h = friction_factor(Re_h, eps, De)
            f_c = friction_factor(Re_c, eps, De)
            Ph[i+1] = Ph[i] - f_h*(dz/De)*(rho_h*v_h**2/2)
            Pc[i+1] = Pc[i] - f_c*(dz/De)*(rho_c*v_c**2/2)
        if abs(Q_total - Q_guess) < tol:
            break
        Q_guess = Q_total
    print_results_full('Iterative', Th, Tc, Ph, Pc, Tw, Q_total, Re_h, Re_c)

# ==============================================================================
# Segmented linear-equation solver: solves 5×5 system per segment sequentially.
# ===============================================================================
def linear_solver():
    Th0, Tc0, Ph0, Pc0, mh, mc, L, A, De, kw, eps, N = interactive_inputs()
    dz    = L / N; A_seg = A / N
    Th = np.zeros(N+1); Tc = np.zeros(N+1)
    Ph = np.zeros(N+1); Pc = np.zeros(N+1); Tw = np.zeros(N)
    Th[0], Tc[0], Ph[0], Pc[0] = Th0, Tc0, Ph0, Pc0
    Q_total = 0.0
    for i in range(N):
        Th_mean = max(Th[i], MIN_TEMP_K); Tc_mean = max(Tc[i], MIN_TEMP_K)
        rho_h, mu_h, Cp_h, k_h, _ = fluid_properties(Th_mean, Ph[i])
        rho_c, mu_c, Cp_c, k_c, _ = fluid_properties(Tc_mean, Pc[i])
        v_h = mh/(rho_h*A_seg); v_c = mc/(rho_c*A_seg)
        hh, Re_h = heat_transfer_coeff(rho_h, v_h, De, mu_h, Cp_h, k_h)
        hc, Re_c = heat_transfer_coeff(rho_c, v_c, De, mu_c, Cp_c, k_c, heating=False)
        Kw = kw*A_seg/De; Au_h=hh*A_seg; Au_c=hc*A_seg
        a = np.array([
            [1, mh*Cp_h,     0,      0,       0],
            [1,-Au_h/2,     Au_h,    0,       0],
            [1,    0,      -Kw,     Kw,      0],
            [1,    0,       0,    -Au_c,   Au_c/2],
            [1,    0,       0,      0,  -mc*Cp_c]
        ], float)
        b = np.array([
            mh*Cp_h*Th[i],
            Au_h/2*Th[i],
            0,
            -Au_c/2*Tc[i],
            -mc*Cp_c*Tc[i]
        ], float)
        x = np.linalg.solve(a, b)
        Q, Th[i+1], Tw[i], _, Tc[i+1] = x; Q_total += Q
        f_h=friction_factor(Re_h, eps, De); f_c=friction_factor(Re_c, eps, De)
        Ph[i+1]=Ph[i]-f_h*(dz/De)*(rho_h*v_h**2/2)
        Pc[i+1]=Pc[i]-f_c*(dz/De)*(rho_c*v_c**2/2)
    U_overall = 1/(1/hh + De/kw + 1/hc)
    print_results_full('Linear', Th, Tc, Ph, Pc, Tw, Q_total, Re_h, Re_c)

# ==============================================================================
# Print segment-wise and summary results
# ===============================================================================
def print_results_full(method, Th, Tc, Ph, Pc, Tw, Q, Re_h, Re_c):
    print(f"\n--- {method} Method Results ---")
    print(f"Outlet Hot Temp: {Th[-1]-273.15:.2f} °C | Outlet Cold Temp: {Tc[-1]-273.15:.2f} °C")
    print(f"Total Heat: {Q:.2f} W | Re_h: {Re_h:.1f}, Re_c: {Re_c:.1f}\n")
    print("Seg | Th(°C) | Tc(°C) | Ph(Pa) | Pc(Pa) | Tw(°C)")
    for i in range(len(Tw)):
        print(f"{i+1:3d} | {Th[i]-273.15:7.2f} | {Tc[i]-273.15:7.2f} | {Ph[i]:8.1f} | {Pc[i]:8.1f} | {Tw[i]-273.15:7.2f}")

# ==============================================================================
# Main entry: select solver method
# ===============================================================================
if __name__ == '__main__':
    print("Select method:\n1) Iterative\n2) Linear")
    choice = input("Choice: ").strip()
    if choice == '1':
        iterative_solver()
    else:
        linear_solver()

# ===============================================================================
# Example usage (inputs):
# Hot inlet temp (°C): 60
# Cold inlet temp (°C): 20
# Hot inlet pressure (Pa): 200000
# Cold inlet pressure (Pa): 100000
# Hot mass flow rate (kg/s): 1.0
# Cold mass flow rate (kg/s): 1.0
# Channel length (m): 1.0
# Total heat transfer area (m²): 2.0
# Equivalent hydraulic diameter (m): 0.02
# Plate conductivity (W/m·K): 15.0
# Surface roughness (m): 1e-5
# Number of segments: 10
# Choice: 1
# ===============================================================================