"""
2D Ray-trace acoustic estimator for six cup shapes (axisymmetric cross-sections).
- Shapes: Cylinder, Cone, Exponential horn, Ellipse (focus->focus), Parabola, Catenary
- Rays are launched from a source location and specularly reflect from the upper/lower walls
  defined by y = ±r(x). The membrane is modeled as a small circular target centered on the axis.
- The script estimates a "geometric gain" from ray counting and converts that to an
  approximate pressure at the membrane center. That pressure is converted to membrane
  displacement using a lumped mechanical impedance.
- This is a geometric-acoustics (ray) approximation — it ignores diffraction/interference.
  Use it for relative comparisons and quick exploration.


"""

import numpy as np
import matplotlib.pyplot as plt


f = 200.0                 # frequency (Hz)
omega = 2 * np.pi * f
p_in = 0.04               # input pressure amplitude (Pa)


R_m = 0.03                        # membrane radius (m)
A_m = np.pi * R_m**2
m_m = 1e-4                        # membrane mass (kg) (total mass)
r_m = 1e-3                        # mechanical damping (Ns/m)
Z_m = 1j * omega * m_m + r_m     # mechanical impedance (force / velocity)


N_RAYS = 4000         # number of rays per shape (increase for smoother stats)
MAX_STEPS = 2000      # per-ray stepping iterations
STEP = 0.005          # marching step (m) along ray
TARGET_RADIUS = 0.01  # membrane detection radius (m) around membrane center
SHOW_SAMPLE_RAYS = False  # set True to show sample rays for each shape (slower)


SOURCE_X = 0.0
SOURCE_Y = 0.0
MEM_X = 0.30          # axial location of membrane (m) for all shapes
MEM_Y = 0.0


X_MAX = 2.0
X_MIN = -0.5
Y_MAX = 1.0
Y_MIN = -1.0


def membrane_displacement_from_pressure(p_mem):
    """Convert pressure amplitude on membrane to center displacement amplitude (m)."""
    # Force = p * A_m
    F = p_mem * A_m
    v = F / Z_m          # velocity (complex)
    disp = np.abs(v) / omega
    return disp

# ----------------------------- Shape definitions -----------------------------
# Each shape gives r(x) (half-height) for x in [0, MEM_X] (or appropriate domain).
# We assume axisymmetric cup opened toward negative x or source at x <= 0 and cup extending toward MEM_X.

def r_cylinder(x, params):
    R = params.get('R', 0.04)
    return np.full_like(x, R)

def r_cone(x, params):
    R1 = params.get('R1', 0.02)   # throat near source
    R2 = params.get('R2', 0.06)   # mouth near membrane
    L = params.get('L', MEM_X)
    return R1 + (R2 - R1) * (x / L)

def r_exponential(x, params):
    R0 = params.get('R0', 0.02)
    R1 = params.get('R1', 0.06)
    L = params.get('L', MEM_X)
    m = np.log(R1 / R0) / L
    return R0 * np.exp(m * x)

def r_ellipse(x, params):
    

    c = MEM_X / 2
    a = 0.16
    b = np.sqrt(a*a - c*c)


    # Place left focus at source and right focus at membrane
    focus_left = SOURCE_X
    focus_right = MEM_X

    # Ellipse center between them
    x_center = (focus_left + focus_right) / 2

    # local coordinate
    X = x - x_center

    # Outside ellipse domain → return zero radius
    inside = 1 - (X**2)/(a**2)

    if isinstance(inside, np.ndarray):
        inside[inside < 0] = 0
    elif inside < 0:
        inside = 0

    return b * np.sqrt(inside)

def r_parabola(x, params):
    # Parabola y^2 = 4 f (x0 - x) opened toward decreasing x with focus near source.
    fpara = params.get('f', 0.10)
    # We'll place the parabola vertex at x_v such that the parabola mouth reaches MEM_X at some half-height.
    # Simplest: vertex at x_v = MEM_X - 0.5*fpara*10  (tunable)
    x_v = params.get('x_v', MEM_X - 0.25)
    # to avoid imaginary values, shift Xloc = x_v - x (we want positive)
    Xloc = x_v - x
    val = 4.0 * fpara * Xloc
    val = np.maximum(val, 0.0)
    return np.sqrt(val)

def r_catenary(x, params):
    a = params.get('a', 0.05)
    # center the catenary so that it's defined around [0, MEM_X]
    x0 = params.get('x0', MEM_X / 2.0)
    Xloc = x - x0
    return a * np.cosh(Xloc / a)

# map names to functions and default params
SHAPES = {
    'Cylinder':   (r_cylinder, {'R': 0.04}),
    'Cone':       (r_cone, {'R1': 0.02, 'R2': 0.06, 'L': MEM_X}),
    'Exponential Horn': (r_exponential, {'R0': 0.02, 'R1': 0.06, 'L': MEM_X}),
    'Ellipse (focus->focus)': (r_ellipse, {'a': 0.15, 'b': 0.10}),
    'Parabola':   (r_parabola, {'f': 0.10, 'x_v': MEM_X - 0.25}),
    'Catenary':   (r_catenary, {'a': 0.05, 'x0': MEM_X / 2.0}),
}

# ----------------------------- Ray utility functions -----------------------------
def sign(x):
    return np.sign(x) if np.abs(x) > 1e-12 else 1.0

def reflect_vector(v, nx, ny):
    n = np.array([nx, ny], dtype=float)
    n = n / np.linalg.norm(n)
    v = np.array(v, dtype=float)
    return v - 2.0 * np.dot(v, n) * n

def find_intersection_with_wall(p_prev, p_curr, r_func, params, axis='upper'):
    """
    Use bisection between p_prev and p_curr to find intersection point with y = ±r(x).
    Returns intersection point and normal (nx, ny) of wall at that x.
    """
    t0 = 0.0
    t1 = 1.0
    x0, y0 = p_prev
    x1, y1 = p_curr
    for _ in range(30):
        tm = 0.5 * (t0 + t1)
        xm = x0 + tm * (x1 - x0)
        ym = y0 + tm * (y1 - y0)
        r_val = r_func(np.array([xm]), params)[0]
        wall_y = r_val if axis == 'upper' else -r_val
        if (ym - wall_y) * (y0 - (r_func(np.array([x0]), params)[0] if axis == 'upper' else -r_func(np.array([x0]), params)[0])) >= 0:
            t0 = tm
        else:
            t1 = tm
    xi = x0 + t1 * (x1 - x0)
    yi = y0 + t1 * (y1 - y0)
    # compute derivative dr/dx numerically for normal direction
    eps = 1e-6
    r_left = r_func(np.array([xi - eps]), params)[0]
    r_right = r_func(np.array([xi + eps]), params)[0]
    drdx = (r_right - r_left) / (2 * eps)
    # wall normal for upper boundary (approx): (-dr/dx, 1)
    nx = -drdx
    ny = 1.0
    if axis == 'lower':
        nx = drdx
        ny = -1.0
    return np.array([xi, yi]), (nx, ny)


def run_raytrace(r_func, params, N=N_RAYS, verbose=False):
    rng = np.random.default_rng(seed=12345)
    # Launch rays isotropically into +x hemisphere from source point
    angles = rng.uniform(-np.pi/2, np.pi/2, N)
    hits = 0
    hit_positions = []
    sample_rays = []  # store small subset for plotting
    for i, theta in enumerate(angles):
        dx = np.cos(theta)
        dy = np.sin(theta)
        pos = np.array([SOURCE_X, SOURCE_Y], dtype=float)
        dirv = np.array([dx, dy], dtype=float)
        prev_pos = pos.copy()
        for step in range(MAX_STEPS):
            prev_pos = pos.copy()
            pos = pos + dirv * STEP
            x, y = pos
            # termination checks
            if x < X_MIN or x > X_MAX or y < Y_MIN or y > Y_MAX:
                break
            # check hit membrane (target circle centered at MEM_X, MEM_Y)
            if (pos[0] >= MEM_X - 0.02) and (np.hypot(pos[0] - MEM_X, pos[1] - MEM_Y) <= TARGET_RADIUS):
                hits += 1
                hit_positions.append(pos.copy())
                break
            # check wall collision: y >= r(x) or y <= -r(x)
            r_val = r_func(np.array([x]), params)[0]
            if r_val == 0: 
                break

            if y >= r_val:
                inter_pt, normal = find_intersection_with_wall(prev_pos, pos, r_func, params, axis='upper')
                nx, ny = normal
                dirv = reflect_vector(dirv, nx, ny)
                pos = inter_pt + 1e-6 * dirv
            elif y <= -r_val:
                inter_pt, normal = find_intersection_with_wall(prev_pos, pos, r_func, params, axis='lower')
                nx, ny = normal
                dirv = reflect_vector(dirv, nx, ny)
                pos = inter_pt + 1e-6 * dirv
        # save a few rays for plotting
        if SHOW_SAMPLE_RAYS and (i % max(1, N // 300) == 0):
            sample_rays.append((angles[i], hit_positions[-1] if hit_positions else None))
    # fraction of rays hitting membrane target
    hit_frac = hits / float(N)
    # compute geometric baseline fraction (if no reflector): approximate angular fraction seen by membrane from source:
    dist = np.hypot(MEM_X - SOURCE_X, MEM_Y - SOURCE_Y)
    if dist <= 0:
        frac_no_reflect = 0.0
    else:
        # angular width of target (2D): theta_t = 2 * arcsin(R_t / dist)
        theta_t = 2.0 * np.arcsin(min(TARGET_RADIUS / dist, 0.9999))
        frac_no_reflect = theta_t / (2.0 * np.pi)
    frac_no_reflect = max(frac_no_reflect, 1e-12)
    geometric_gain = (hit_frac / frac_no_reflect) if frac_no_reflect > 0 else 0.0
    # convert geometric_gain to pressure amplification (pressure ~ sqrt(intensity) ~ sqrt(gain))
    # if geometric_gain < 1, we still take sqrt; if zero, pressure is effectively zero.
    p_mem_est = p_in * np.sqrt(max(geometric_gain, 0.0))
    disp = membrane_displacement_from_pressure(p_mem_est)
    result = {
        'hits': hits,
        'hit_frac': hit_frac,
        'frac_no_reflect': frac_no_reflect,
        'geometric_gain': geometric_gain,
        'p_mem': p_mem_est,
        'disp': disp,
        'hit_positions': np.array(hit_positions),
        'sample_rays': sample_rays
    }
    if verbose:
        print(f"Hits: {hits}/{N}, hit_frac={hit_frac:.4e}, frac_no_reflect={frac_no_reflect:.4e}, gain={geometric_gain:.3e}")
        print(f"Estimated p_mem = {p_mem_est:.6e} Pa, disp = {disp:.6e} m")
    return result


results = {}
for name, (rfunc, params) in SHAPES.items():
    print(f"Running raytrace for: {name}")
    res = run_raytrace(rfunc, params, N=N_RAYS, verbose=True)
    results[name] = res

names = list(results.keys())
disps = [results[n]['disp'] for n in names]
p_mems = [results[n]['p_mem'] for n in names]

print("\n=== SUMMARY ===")
for n in names:
    print(f"{n:25s} : displacement = {results[n]['disp']:.4e} m, p_mem = {results[n]['p_mem']:.4e} Pa, hits = {results[n]['hits']}")

plt.figure(figsize=(10,5))
plt.bar(names, disps)
plt.ylabel("Membrane center displacement (m)")
plt.title(f"Estimated steady-state displacement (f={f} Hz, p_in={p_in} Pa)")
plt.xticks(rotation=25)
plt.grid(axis='y')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,5))
plt.bar(names, p_mems)
plt.ylabel("Estimated pressure at membrane (Pa)")
plt.title("Estimated pressure at membrane (geometric ray model)")
plt.xticks(rotation=25)
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# Optional: plot hit positions for each shape
for name in names:
    hits = results[name]['hit_positions']
    if hits.size == 0:
        continue
    plt.figure(figsize=(6,3))
    plt.scatter(hits[:,0], hits[:,1], s=1)
    # plot membrane center
    circle = plt.Circle((MEM_X, MEM_Y), TARGET_RADIUS, color='red', fill=False)
    plt.gca().add_patch(circle)
    plt.title(f"Hit positions (sample) - {name}")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.axis('equal')
    plt.grid(True)
    plt.show()
