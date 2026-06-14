import numpy as np
import matplotlib.pyplot as plt


R = 0.1
T = 1.0
rho = 0.01
c = np.sqrt(T / rho)
gamma = 5.0

p0 = 0.04
z = 1.0

frequencies = [100,120,140,160,180,200,220,240,260,280,300,320,340,360,380,400]

Nr = 200
dr = R / (Nr - 1)
r = np.linspace(0, R, Nr)

dt = 0.35 * dr / c
t_end = 1
Nt = int(t_end / dt)

inv_dr2 = 1.0 / dr**2
inv_2dr = 1.0 / (2*dr)


def radial_operator(u):
    L = np.zeros_like(u)
    for i in range(1, Nr-1):
        d2 = (u[i+1] - 2*u[i] + u[i-1]) * inv_dr2
        d1 = (u[i+1] - u[i-1]) * inv_2dr
        L[i] = d2 + (1/r[i]) * d1
    L[0] = 2 * (u[1] - u[0]) * inv_dr2
    return L


steady_amplitudes = []

for f in frequencies:
    omega = 2*np.pi*f
    
    def accel(t):
        # pressure wave
        pressure_acc = (p0 / z) * np.sin(omega * t) / rho

        # point force at center
        F_point = 0.5
        point_acc = F_point / (dr**2) / rho

        a = np.full(Nr, pressure_acc)
        a[0] += point_acc
        return a

    w = np.zeros(Nr)
    w_prev = np.zeros(Nr)
    center_hist = []

    for n in range(Nt):
        t = n * dt
        Lw = radial_operator(w)

        w_next = (
            2*w - (1 - gamma*dt)*w_prev + dt**2 * (c**2 * Lw + accel(t))
        ) / (1 + gamma*dt)

        w_next[-1] = 0.0

        w_prev, w = w, w_next
        center_hist.append(w[0])

    # Extract steady-state amplitude
    steady_signal = center_hist[int(0.9*len(center_hist)):]
    amplitude = 0.5 * (np.max(steady_signal) - np.min(steady_signal))
    steady_amplitudes.append(amplitude)

# Which frequency gives max oscillation?
idx_max = np.argmax(steady_amplitudes)
best_freq = frequencies[idx_max]
print("Frequency with maximum oscillation =", best_freq, "Hz")



omega = 2*np.pi*best_freq

def accel_best(t):
    pressure_acc = (p0 / z) * np.sin(omega * t) / rho
    F_point = 0.5
    point_acc = F_point / (dr**2) / rho
    a = np.full(Nr, pressure_acc)
    a[0] += point_acc
    return a

w = np.zeros(Nr)
w_prev = np.zeros(Nr)
w_store = []  # store radial profiles near steady state

for n in range(Nt):
    t = n * dt
    Lw = radial_operator(w)
    w_next = (
        2*w - (1 - gamma*dt)*w_prev + dt**2 * (c**2 * Lw + accel_best(t))
    ) / (1 + gamma*dt)

    w_next[-1] = 0.0

    w_prev, w = w, w_next

    # store last 10% of time steps for steady state profile
    if n > 0.9*Nt:
        w_store.append(w.copy())

# Steady-state radial displacement = last stored profile
w_final = w_store[-1]



plt.figure(figsize=(8,5))
plt.plot(r, w_final, linewidth=2)
plt.xlabel("Radius r (m)")
plt.ylabel("Displacement w (m)")
plt.title(f"Steady-State Displacement vs Radius at {best_freq} Hz")
plt.grid(True)
plt.show()

