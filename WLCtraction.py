import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid

def get_shape_from_coeffs(coeffs_real, coeffs_imag, L, s_points):
 
    ## Reconstructs the curvature (k), theta and the (x,y) coordinates from the Fourier coefficients. ##
    
    k = np.zeros_like(s_points)
    # Adds the constant (mode 0)
    k += coeffs_real[0]
    
    # Adds the higher modes
    for n in range(1, len(coeffs_real)):
        qn = 2 * np.pi * n / L
        k += 2 * coeffs_real[n] * np.cos(qn * s_points) - 2 * coeffs_imag[n] * np.sin(qn * s_points)
        
    theta = cumulative_trapezoid(k, s_points, initial=0)
    x = cumulative_trapezoid(np.cos(theta), s_points, initial=0)
    y = cumulative_trapezoid(np.sin(theta), s_points, initial=0)
    
    return k, x, y

def calculate_energy(k, x_end, lp, f, ds):
   
    ## Calculates total energy E = U - FX. ##
    # bending Energy U = (lp/2) * integral(k^2 ds)
    # Note: EI/kT is in lp. We assume kT=1 for the simulation.
    bending_energy = (lp / 2.0) * np.sum(k**2) * ds
    
    # potential energy of the external force V = -F * X
    force_energy = -f * x_end
    
    return bending_energy + force_energy

def simulate_filament_with_force(L, lp, f, N_modes=15, steps=5000):
   
   ## Simulates the filamente using Metropolis-Hastings on the Fourier coefficients ##
   ## f: Force (normalized for kT). f=0 -> random coil, f big -> strained filament. ##
 
    s = np.linspace(0, L, 1000)
    ds = s[1] - s[0]
    
    # coefficients inizialization
    sig = np.sqrt(L / lp)
    # memorizing the coefficients in an array
    coeffs_real = np.zeros(N_modes)
    coeffs_imag = np.zeros(N_modes)
    
    # Initial random state
    coeffs_real[:] = np.random.randn(N_modes) * sig * 0.1 # starting with low curvature
    coeffs_imag[:] = np.random.randn(N_modes) * sig * 0.1

    # Inital energy calculation
    k, x, y = get_shape_from_coeffs(coeffs_real, coeffs_imag, L, s)
    current_E = calculate_energy(k, x[-1], lp, f, ds)
    
    accepted = 0
    
    # Metropolis loop
    print(f"Simulation running (Forza F={f})...")
    for step in range(steps):
        # 1. modifying a random coefficient
        idx = np.random.randint(0, N_modes)
        is_real = np.random.rand() > 0.5
        delta = np.random.randn() * 1.0 # perturbation amplitude
        
        # saving the old value
        old_val = coeffs_real[idx] if is_real else coeffs_imag[idx]
        
        # Perturbation
        if is_real:
            coeffs_real[idx] += delta
        else:
            coeffs_imag[idx] += delta
            
        # 2. calculating the new shape and new energy
        k_new, x_new, y_new = get_shape_from_coeffs(coeffs_real, coeffs_imag, L, s)
        new_E = calculate_energy(k_new, x_new[-1], lp, f, ds)
        
        # 3. metropolis: we accept the value if the energy decreases or the probability is e^(-dE)
        delta_E = new_E - current_E
        
        if delta_E < 0 or np.random.rand() < np.exp(-delta_E):
            # accepts
            current_E = new_E
            accepted += 1
        else:
            # denies
            if is_real:
                coeffs_real[idx] = old_val
            else:
                coeffs_imag[idx] = old_val

    # Final shape
    k_final, x_final, y_final = get_shape_from_coeffs(coeffs_real, coeffs_imag, L, s)
    return x_final, y_final, accepted/steps

# --- parameters ---
L = 3.0       # total length
lp = 1.0      # persistence length
f_val = 1.0   # force

# Simulation
x, y, acc_rate = simulate_filament_with_force(L, lp, f=f_val, steps=2000)

# Plot
plt.figure(figsize=(6, 8))
plt.plot(x, y, label=fr'Filament ($\xi$={lp}, L={L})', lw=2)

# Drawing force
plt.arrow(x[-1], y[-1], 0.5, 0, head_width=0.1, head_length=0.1, fc='r', ec='r', label=fr'F={f_val}')
plt.text(x[-1]+0.6, y[-1], 'F', color='red', fontsize=14)

# graphic setup
plt.axvline(0, color='k', linestyle='--', alpha=0.3) # wall
plt.scatter([0], [0], color='black', zorder=5) # anchor point
plt.xlabel('x (Force direction)')
plt.ylabel('y')
plt.title(f'Worm-like Chain under traction (Rate acc.: {acc_rate:.2f})')
plt.axis('equal')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(f"WLCuT{f_val}.png", dpi=300, bbox_inches='tight')
plt.show()