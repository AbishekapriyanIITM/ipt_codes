import numpy as np
import matplotlib.pyplot as plt


R = 0.1                 # radius of membrane (m)
T = 1.0                 # tension (N/m)
rho = 0.01              # surface density (kg/m^2)
c = np.sqrt(T / rho)    # wave speed

p0 = 1.0                # pressure amplitude (Pa)
z = 1.0                 # distance of sound source (m)
f = 200.0               # frequency of sound (Hz)
omega = 2*np.pi*f

# time at which profile is required
t_plot = 7         # seconds  


Nr = 150
dr = R / (Nr - 1)
r = np.linspace(0, R, Nr)

dt = 0.4 * dr / c
t_total = t_plot
Nt = int(t_total / dt)


def pressure(t):
    return (p0 / z) * np.sin(omega * t)

def accel(t):
    return pressure(t) / rho


w_prev = np.zeros(Nr)
w = np.zeros(Nr)

inv_dr2 = 1.0 / dr**2
inv_2dr = 1.0 / (2*dr)

# Radial Laplacian
def radial_operator(u):
    L = np.zeros_like(u)
    for i in range(1, Nr-1):
        d2 = (u[i+1] - 2*u[i] + u[i-1]) * inv_dr2
        d1 = (u[i+1] - u[i-1]) * inv_2dr
        L[i] = d2 + (1/r[i]) * d1
    L[0] = 2*(u[1] - u[0]) * inv_dr2  # center condition
    return L


for n in range(Nt):
    t = n * dt
    Lw = radial_operator(w)
    w_next = 2*w - w_prev + dt**2 * (c**2 * Lw + accel(t))
    
    w_next[-1] = 0.0   # clamped edge condition
    w_prev, w = w, w_next


plt.figure(figsize=(8,5))
plt.plot(r, w, 'b', linewidth=2)
plt.xlabel("Radial distance r (m)")
plt.ylabel("Membrane displacement w (m)")
plt.title(f"Membrane Oscillation Profile at t = {t_plot:.5f} s")
plt.grid(True)
plt.show()


