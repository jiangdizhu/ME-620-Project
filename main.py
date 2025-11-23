import numpy as np
import matplotlib.pyplot as plt


class CoupledVocalFoldSimulation:
    def __init__(self, num_slices=20, dt=1e-5, total_time=0.02, E_L=40000.0, label="Healthy"):
        """
        2.5D Simulation with Longitudinal Coupling.
        Allows customizing Longitudinal Stiffness (E_L) to simulate paralysis.
        """
        self.label = label
        self.dt = dt
        self.num_steps = int(total_time / dt)
        self.time = np.linspace(0, total_time, self.num_steps)
        self.num_slices = num_slices
        self.L_vf = 0.015 
        self.dy = self.L_vf / num_slices 

        # Material Properties
        self.rho = 1040.0
        self.E_t = 5000.0      # Transverse Stiffness (remains similar)
        self.E_L = E_L         # Longitudinal Stiffness (Variable)
        self.nu_t = 0.4
        self.nu_L = 0.3
        
        # Longitudinal Shear Modulus (mu') scales with E_L in this simplified model
        # assumption: Shear stiffness drops with muscle tone loss
        self.mu_L = E_L / 2.5 

        # Derived Transverse Shear Modulus (mu)
        self.mu = self.E_t / (2 * (1 + self.nu_t))

        # Stiffness Coefficients c1, c2
        self.alpha = self.E_L / self.E_t
        denom = self.alpha * (1 - self.nu_t) - 2 * self.nu_L**2
        self.c1 = (2 * (self.alpha - self.nu_L**2)) / denom
        
        self.subglottal_pressure = 800.0 

    def element_matrices(self, coords):
        """ Calculate 2D element matrices for a single slice. """
        x, z = coords[:, 0], coords[:, 1]
        b = np.array([z[1]-z[2], z[2]-z[0], z[0]-z[1]])
        c = np.array([x[2]-x[1], x[0]-x[2], x[1]-x[0]])
        Area = 0.5 * np.abs(x[0]*(z[1]-z[2]) + x[1]*(z[2]-z[0]) + x[2]*(z[0]-z[1]))
        
        # Stiffness (K)
        K_el = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                val = (self.mu * self.dy / (4 * Area)) * (self.c1 * b[i] * b[j] + c[i] * c[j])
                K_el[i, j] = val
        
        # Mass (M)
        M_el = np.zeros((3, 3))
        const_mass = (self.rho * Area * self.dy) / 12.0
        for i in range(3):
            M_el[i, i] = 2 * const_mass
            for j in range(i+1, 3):
                M_el[i, j] = M_el[j, i] = const_mass
                
        return K_el, M_el, Area

    def run_simulation(self):
        coords = np.array([[0.0, 0.0], [0.005, 0.0], [0.0, 0.005]]) 
        K_slice, M_slice, Area = self.element_matrices(coords)
        
        # Damping (Rayleigh)
        D_slice = 0.001 * K_slice + 1.0 * M_slice 
        M_inv = np.linalg.inv(M_slice)

        active_slices = range(1, self.num_slices - 1)
        U = np.zeros((self.num_slices, 3)) 
        U_prev = np.zeros((self.num_slices, 3))
        history = np.zeros((self.num_steps, self.num_slices))

        for t_idx, t in enumerate(self.time):
            F_net = np.zeros((self.num_slices, 3))
            
            for k in active_slices:
                # Aerodynamic Force (Same driver for both to isolate material effect)
                if 5 <= k <= 15 and t < 0.015: 
                    force_mag = self.subglottal_pressure * 0.00005 * np.sin(2*np.pi*200*t)
                    F_net[k, 1] += force_mag 
                
                f_elastic = -np.dot(K_slice, U[k])
                F_net[k] += f_elastic

                vel = (U[k] - U_prev[k]) / self.dt
                f_damping = -np.dot(D_slice, vel)
                F_net[k] += f_damping

                # String Forces (Longitudinal)
                k_long = (self.mu_L * Area) / self.dy 
                f_string = k_long * (U[k+1] - U[k]) + k_long * (U[k-1] - U[k])
                F_net[k] += f_string

            U_next = np.zeros_like(U)
            for k in active_slices:
                acc = np.dot(M_inv, F_net[k])
                U_next[k] = 2*U[k] - U_prev[k] + acc * self.dt**2
            
            U_next[0] = 0
            U_next[-1] = 0
            U_prev = U.copy()
            U = U_next.copy()
            history[t_idx, :] = U[:, 1]

        return self.time, history

# Run
sim = CoupledVocalFoldSimulation()
t, data = sim.run_simulation()

# Plot
plt.figure(figsize=(10, 6))
# Create a heatmap: X-axis = Slices (Anterior-Posterior), Y-axis = Time
plt.imshow(data.T, aspect='auto', extent=[0, 0.02, 0, 15], origin='lower', cmap='RdBu_r')
plt.colorbar(label='Displacement (m)')
plt.title('Vocal Fold Vibration: Longitudinal View (Top-Down)')
plt.xlabel('Time (s)')
plt.ylabel('Slice Index (Posterior -> Anterior)')
plt.show()

# --- Compare Healthy vs Paralyzed ---

# 1. Healthy Fold (High Tension/Stiffness)
# E_L = 40 kPa (Active Muscle)
sim_healthy = CoupledVocalFoldSimulation(E_L=40000.0, label="Healthy")
t, data_healthy = sim_healthy.run_simulation()

# 2. Paralyzed Fold (Loss of Tension/Stiffness)
# E_L = 4 kPa (Passive/Atrophied Muscle - 10x reduction)
# According to report: "lowering the longitudinal stiffness E'"
sim_paralyzed = CoupledVocalFoldSimulation(E_L=4000.0, label="Paralyzed")
t, data_paralyzed = sim_paralyzed.run_simulation()

# --- Visualization ---
# Plot Center Slice (Slice 10) for direct comparison
plt.figure(figsize=(10, 6))
plt.plot(t, data_healthy[:, 10] * 1000, 'b-', linewidth=2, label='Healthy Fold ($E_L=40$ kPa)')
plt.plot(t, data_paralyzed[:, 10] * 1000, 'r--', linewidth=2, label='Paralyzed Fold ($E_L=4$ kPa)')

plt.title("Comparison of Vocal Fold Vibration: Healthy vs. Paralyzed (Center Slice)")
plt.xlabel("Time (s)")
plt.ylabel("Lateral Displacement (mm)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# Plot Heatmap of Paralyzed Fold to show "Flaccid" behavior
plt.figure(figsize=(10, 6))
plt.imshow(data_paralyzed.T, aspect='auto', extent=[0, 0.02, 0, 15], origin='lower', cmap='RdBu_r', vmin=np.min(data_healthy), vmax=np.max(data_healthy))
plt.colorbar(label='Displacement (m)')
plt.title('Paralyzed Fold Vibration (Top-Down View)')
plt.xlabel('Time (s)')
plt.ylabel('Slice Index')
plt.show()