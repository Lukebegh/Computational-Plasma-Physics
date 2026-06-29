# Constant Coefficient Advection

A comparison of five numerical schemes for solving the 1D linear advection equation on a periodic domain, implemented in Python. The schemes range from first-order finite differences to a spectrally accurate Fourier method, and are benchmarked against each other by tracking the mass and L²-norm of the solution over time.

---

## The Problem

The 1D advection equation transports a scalar field f(t, x) at constant speed v:

$$
\partial_t f + v \partial_x f = 0
$$

on the periodic domain [0, 2π]. The exact solution is a rigid translation of the initial profile — any departure from this introduced by the numerics reveals the dissipation, dispersion, or instability of the scheme. The initial condition is a Gaussian pulse centred in the domain:

$$
f(0, x) = \frac{1}{\sqrt{2\pi}\,\sigma} \exp\left(-\frac{(x - \pi)^2}{2\sigma^2}\right)
$$

---

## Code Structure

```
constant coefficient advection/
├── main.py                     # Configuration, setup, time-stepping, and plotting
├── Explicit_upwind_scheme.py   # First-order explicit upwind
├── Implicit_upwind_scheme.py   # First-order implicit upwind
├── Lax_Wendroff_scheme.py      # Second-order explicit Lax–Wendroff
├── Crank_Nicholson_scheme.py   # Second-order implicit Crank–Nicolson
└── Pseudospectral_method.py    # Fourier pseudospectral method
```

Each scheme lives in its own module and exposes a single function that takes the physical parameters, grid spacings, initial condition, and number of time steps, and returns the full solution matrix `F` of shape `(p+1, N)`, where rows are time steps and columns are spatial grid points. `main.py` imports whichever scheme is selected, runs it, computes the diagnostics, and plots the result.

---

## Numerical Methods

All five finite-difference schemes are cast as matrix-vector multiplications of the form F<sup>n+1</sup> = C F<sup>n</sup>, making the time-stepping loop in each module a single `np.matmul` call per step.

### Explicit Upwind (`Explicit_upwind_scheme.py`)

The propagation matrix A has diagonal entries `1 - ν` and sub-diagonal entries `ν`, where `ν = v Δt / Δx` is the Courant number, with periodic wrap-around enforced in the corner entry. This is first-order accurate in both space and time and is conditionally stable: the CFL condition `0 ≤ ν ≤ 1` must be satisfied, setting a maximum allowable time step of `Δt ≤ Δx / v`. Exceeding this causes the solution to blow up. For v < 0 the upwind direction flips, and the scheme as written (which differences to the left) becomes unstable.

### Implicit Upwind (`Implicit_upwind_scheme.py`)

The same upwind spatial stencil is applied at the new time level, giving an implicit system A F<sup>n+1</sup> = F<sup>n</sup>, where A has diagonal entries `1 + ν` and sub-diagonal entries `-ν`. Each step solves this via `np.linalg.inv(A)`, applied once per step. The scheme is unconditionally stable — any Δt keeps the solution bounded — though large time steps still degrade temporal accuracy. It is also first-order accurate.

### Lax–Wendroff (`Lax_Wendroff_scheme.py`)

The propagation matrix A has diagonal entries `1 - ν²`, sub-diagonal entries `ν(ν+1)/2`, and super-diagonal entries `ν(ν-1)/2`. This second-order correction eliminates the leading dissipation error of the upwind scheme. It is conditionally stable under the same CFL condition `|ν| ≤ 1`, but introduces dispersive rather than dissipative errors — visible as oscillatory ringing behind sharp features.

### Crank–Nicolson (`Crank_Nicholson_scheme.py`)

The centred spatial derivative is averaged between the current and next time level. This gives an implicit system A F<sup>n+1</sup> = B F<sup>n</sup>, where A has diagonal entries `1` and off-diagonal entries `±ν/4`, and B is its mirror. The combined step matrix C = A<sup>-1</sup>B is precomputed once before the loop. The scheme is second-order accurate, unconditionally stable, and — crucially — non-dissipative: the amplification factor has magnitude exactly 1 for all wavenumbers, so the L²-norm is preserved.

### Fourier Pseudospectral (`Pseudospectral_method.py`)

Rather than a matrix method, this approach works directly in frequency space. The initial condition is transformed via `np.fft.fft`, and each time step applies an exact phase rotation to every Fourier mode:

$$
\hat{f}_k^{n+1} = e^{-2\pi i k v \Delta t / L} \hat{f}_k^n
$$

The physical-space solution is recovered via `np.fft.ifft`. Because the time evolution of each mode is solved exactly, the method is spectrally accurate — errors decay exponentially with N for smooth data — and conserves both mass and L²-norm to machine precision. There is no CFL restriction beyond the requirement that the solution remains well-resolved on the grid.

---

## Diagnostics

After the time-stepping loop, `main.py` computes two integral quantities at each time step using the rectangle-rule approximation (summing Δx × f over all grid points):

- **Mass** m(t) = ∫ f(t,x) dx — should be conserved by all schemes
- **L²-norm** ‖f‖²(t) = ∫ |f(t,x)|² dx — decays in dissipative schemes, conserved in non-dissipative ones

These reveal the qualitative character of each method without needing to compare against an analytic solution at every point.

---

## Usage

All parameters are set at the top of `main.py`:

| Parameter | Variable | Default |
|-----------|----------|---------|
| Advection speed | `v` | `2` |
| Spatial domain | `a`, `b` | `0`, `2π` |
| Number of grid points | `N` | `100` |
| Number of time steps | `p` | `160` |

Select a scheme by setting `method` to one of:

| Value | Scheme |
|-------|--------|
| `1` | Explicit upwind |
| `2` | Implicit upwind |
| `3` | Lax–Wendroff |
| `4` | Crank–Nicolson |
| `5` | Pseudospectral |

Set `graph = 1` to plot the mass or `graph = 2` to plot the L²-norm, then run:

```bash
python main.py
```

---

## Dependencies

- Python 3.x
- NumPy
- Matplotlib

```bash
pip install numpy matplotlib
```
