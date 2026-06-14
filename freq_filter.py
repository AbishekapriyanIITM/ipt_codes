import numpy as np
import matplotlib.pyplot as plt
import time


fs = 25000             
dt = 1.0 / fs
T_total = 2.0          # seconds
Nt = int(np.round(T_total * fs))
t = np.linspace(0, T_total, Nt)

print(f"fs={fs}, dt={dt:.3e}, Nt={Nt}")


input_freqs = np.arange(100, 1001, 50)
input_signal = np.zeros(Nt, dtype=float)
for f in input_freqs:
    input_signal += np.sin(2*np.pi*f*t)
# normalize
input_signal /= np.max(np.abs(input_signal))


Nx, Ny = 40, 40
dx = 0.01
c_mem = 120.0

Ns = 120
dx_s = 0.02
c_s = 200.0

# small damping (velocity-proportional)
gamma_mem = 5e-3
gamma_str = 1e-3

# Courant check (informational)
C_mem = c_mem * dt / dx
C_str = c_s * dt / dx_s
print(f"C_mem = {C_mem:.6f}, C_str = {C_str:.6f}  (should be << ~0.7)")


u = np.zeros((Nx, Ny), dtype=float)        # membrane1 current
u_prev = np.zeros_like(u)
u_next = np.zeros_like(u)

u2 = np.zeros_like(u)                      # membrane2
u2_prev = np.zeros_like(u)
u2_next = np.zeros_like(u)

s = np.zeros(Ns, dtype=float)
s_prev = np.zeros_like(s)
s_next = np.zeros_like(s)

center_idx = (Nx//2, Ny//2)

mem1_center = np.empty(Nt, dtype=float)
mem2_center = np.empty(Nt, dtype=float)


coef_mem = (c_mem * dt / dx)**2
coef_str = (c_s * dt / dx_s)**2


start_time = time.time()
abort_threshold = 1e4   # safety clamp threshold


for n in range(Nt):
    # Left membrane update (vectorized interior)
    # laplacian via slicing
    u_next[1:-1,1:-1] = (
        2.0*u[1:-1,1:-1] - u_prev[1:-1,1:-1]
        + coef_mem * (u[2:,1:-1] + u[:-2,1:-1] + u[1:-1,2:] + u[1:-1,:-2] - 4.0*u[1:-1,1:-1])
        - gamma_mem * (u[1:-1,1:-1] - u_prev[1:-1,1:-1])
    )

    # Drive at center (point driver). scale factor 0.05 keeps amplitude moderate
    u_next[center_idx] += input_signal[n] * 0.05

    # Clamped edges (Dirichlet = 0); you can change to free edges by copying neighbors
    u_next[0,:] = 0.0
    u_next[-1,:] = 0.0
    u_next[:,0] = 0.0
    u_next[:,-1] = 0.0

    # record center displacement of left membrane if needed
    mem1_center[n] = u_next[center_idx]

    # String update (vectorized interior)
    s_next[1:-1] = (2.0*s[1:-1] - s_prev[1:-1]
                    + coef_str * (s[2:] - 2.0*s[1:-1] + s[:-2])
                    - gamma_str * (s[1:-1] - s_prev[1:-1]))

    # Drive string left endpoint from membrane center strongly (Dirichlet)
    s_next[0] = u[center_idx]   # you can use u_next[center_idx] or averaged patch for different coupling

    # fix string right end to zero (clamped)
    s_next[-1] = 0.0

    # Right membrane update (vectorized interior)
    u2_next[1:-1,1:-1] = (
        2.0*u2[1:-1,1:-1] - u2_prev[1:-1,1:-1]
        + coef_mem * (u2[2:,1:-1] + u2[:-2,1:-1] + u2[1:-1,2:] + u2[1:-1,:-2] - 4.0*u2[1:-1,1:-1])
        - gamma_mem * (u2[1:-1,1:-1] - u2_prev[1:-1,1:-1])
    )

    # Coupling: drive small patch on right membrane with string end displacement
    # Here simplest: drive center of membrane2 by string (end-1) scaled
    # (you can replace with patch averaging if you want more realistic coupling)
    u2_next[center_idx] += s[-2] * 0.05

    # clamp edges
    u2_next[0,:] = 0.0
    u2_next[-1,:] = 0.0
    u2_next[:,0] = 0.0
    u2_next[:,-1] = 0.0

    mem2_center[n] = u2_next[center_idx]

    # advance time levels
    u_prev, u = u, u_next
    u_next = np.zeros_like(u)   # reuse new array (cleared)
    u2_prev, u2 = u2, u2_next
    u2_next = np.zeros_like(u2)
    s_prev, s = s, s_next
    s_next = np.zeros_like(s)

    # light safety check and progress print
    if (n % max(1, Nt//10)) == 0:
        pct = 100.0 * n / Nt
        print(f"t={n}/{Nt} ({pct:.1f}%)  maxL={np.max(np.abs(u)):.3e}  maxS={np.max(np.abs(s)):.3e}  maxR={np.max(np.abs(u2)):.3e}")

    if (np.max(np.abs(u)) > abort_threshold) or (np.max(np.abs(u2)) > abort_threshold) or (np.max(np.abs(s)) > abort_threshold):
        print("ABORT: amplitude exceeded threshold, stopping early.")
        break

end_time = time.time()
print("Done. elapsed:", end_time - start_time, "s")

def compute_fft(signal, fs):
    N = len(signal)
    X = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(N, d=1/fs)
    return freqs, np.abs(X)

freq_in, fft_in = compute_fft(input_signal, fs)
freq_m2, fft_m2 = compute_fft(mem2_center, fs)
freq_m1, fft_m1 = compute_fft(mem1_center, fs)

# normalize
fft_in /= (fft_in.max() if fft_in.max()!=0 else 1.0)
fft_m2 /= (fft_m2.max() if fft_m2.max()!=0 else 1.0)
fft_m1 /= (fft_m1.max() if fft_m1.max()!=0 else 1.0)

plt.figure(figsize=(10,4))
#plt.plot(freq_in, fft_in, label='Input FFT', alpha=0.6)
plt.plot(freq_m2, fft_m2, label='Membrane2 FFT', alpha=0.8,color='black')
plt.plot(freq_m1, fft_m1, label='Membrane1 FFT', alpha=0.8,color='red')
plt.xlim(0, 1500)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Normalized amplitude")
plt.title("Frequency Domain (FFT) Comparison")
plt.legend()
plt.grid(True)
plt.show()
