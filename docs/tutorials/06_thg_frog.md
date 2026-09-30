---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Python 3
  name: python3
---

# THG-FROG

Third-harmonic-generation FROG gates the pulse with itself through a
$\chi^{(3)}$ process, $E_\mathrm{sig}(t,\tau) = E(t)^2\,E(t-\tau)$, and measures
the signal near $3\omega_0$ — around 267 nm for an 800 nm pulse. It needs no
phase-matched crystal, which makes it a natural choice for few-cycle pulses, but
it brings three things to watch out for:

1. the trace lives at the **third harmonic** and is much **broader** than the
   fundamental, so the retrieval band has to be sized from the trace;
2. its frequency marginal gives only **soft** consistency anchors, centred on
   $3\bar\omega_S$;
3. it fixes the direction of time in general, but it is **blind to the sign of a
   pure linear chirp on a Gaussian pulse**.

This tutorial walks through all three on a synthetic measurement, then retrieves
the pulse. The background is in [Interactions](../explanation/interactions.md#thg-frog)
and [Marginals](../explanation/marginals.md).

```{code-cell} ipython3
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import croak
from croak import marginal_checks as mc
from croak.maths import wlfreq

C = croak.constants.C
```

## A pulse and its THG trace

We build an 800 nm pulse with an 8 fs transform limit, and add both group-delay
dispersion (GDD) and third-order dispersion (TOD). The synthesis grid is
deliberately fine and long, so the "measurement" is not limited by it.

```{code-cell} ipython3
omega0 = wlfreq(800e-9)
g = croak.Grid(1024, dt=0.25e-15)
ew = croak.gaussian_pulse(g, 8e-15, phases=[40e-30, 600e-45])   # GDD 40 fs², TOD 600 fs³
truth = croak.TruthPulse.from_spectrum(g, ew, omega0)
true_fwhm = croak.maths.fwhm(g.t, np.abs(g.ifft(ew)) ** 2)
print(f"true temporal FWHM = {true_fwhm / 1e-15:.2f} fs")

delays = np.linspace(-80e-15, 80e-15, 161)
trace = croak.maketrace(g.omega, delays, ew, "thg")
```

{func}`~croak.maketrace` works on the baseband grid: the trace rows are offsets
from the *signal* carrier, which for THG is
`get_interaction("thg").omega0_scale` $=3$ times the pulse carrier. Adding
$3\omega_0$ puts the trace on its absolute wavelength axis:

```{code-cell} ipython3
scale = croak.get_interaction("thg").omega0_scale
omega_signal = g.omega + scale * omega0
# The wide synthesis grid reaches below zero absolute frequency; keep the
# physical rows and sort them so the wavelength axis is monotonic.
physical = omega_signal > 0
lam_signal = wlfreq(omega_signal[physical])
order = np.argsort(lam_signal)

fig, ax = plt.subplots(figsize=(5.5, 4))
ax.pcolormesh(delays / 1e-15, lam_signal[order] / 1e-9, trace[physical][order],
              shading="auto")
ax.set_ylim(235, 305)
ax.set_xlabel("delay (fs)")
ax.set_ylabel("signal wavelength (nm)")
ax.set_title("THG-FROG trace of an 800 nm pulse")
fig.tight_layout()
```

The TOD makes the trace visibly asymmetric in delay: THG treats its two arms
differently, so unlike SHG it tells a pulse from its time-reversed copy.

## The chirp-sign blind spot

That is not the whole story. For a **Gaussian with pure GDD**, $E(t)^2E(t-\tau)$
is a Gaussian times a linear chirp, both centred on $t=\tau/3$. The trace then
depends on $\tau$ only through $\tau^2$, so $+$GDD and $-$GDD give *identical*
THG traces. SD-FROG, by contrast, sees the sign, and adding TOD restores it for
THG:

```{code-cell} ipython3
def trace_of(interaction, phases):
    pulse = croak.gaussian_pulse(g, 8e-15, phases=phases)
    return croak.maketrace(g.omega, delays, pulse, interaction)

cases = {
    "THG, ±40 fs² GDD": ("thg", [40e-30], [-40e-30]),
    "SD, ±40 fs² GDD": ("sd", [40e-30], [-40e-30]),
    "THG, ±40 fs² GDD + 600 fs³ TOD": ("thg", [40e-30, 600e-45], [-40e-30, 600e-45]),
}
fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
for ax, (label, (inter, plus, minus)) in zip(axes, cases.items()):
    diff = trace_of(inter, plus) - trace_of(inter, minus)
    print(f"{label:32s} max |ΔT| = {np.abs(diff).max():.1e}")
    ax.pcolormesh(delays / 1e-15, g.omega / 1e15, diff, cmap="RdBu_r",
                  vmin=-0.5, vmax=0.5, shading="auto")
    ax.set_ylim(-1.5, 1.5)
    ax.set_title(label, fontsize=9)
    ax.set_xlabel("delay (fs)")
axes[0].set_ylabel(r"$\omega - \omega_\mathrm{sig}$ (rad/fs)")
fig.suptitle("trace(+GDD) − trace(−GDD)")
fig.tight_layout()
```

The first panel is blank to machine precision. On a real, near-Gaussian pulse
dominated by GDD, the chirp sign is therefore encoded only weakly. Confirm it
independently (for instance with a known glass insertion) when it matters. It
also means a **flat-phase starting guess sits on a saddle** between the two
mirror solutions, where a solver converges slowly. Starting from a guess with a
little chirp, of either sign, avoids that. The default `guess="auto"` used below
builds its phase from the trace's per-frequency delay centroid, which the TOD
makes non-trivial here. On a delay-symmetric trace that phase comes out flat, so
seed a small chirp by hand instead (pass `guess=` a chirped spectrum).

## A synthetic measurement

A spectrometer records a *wavelength* density. We convert the trace, crop it to a
UV spectrometer's range and add a little detector noise. We also record the
fundamental spectrum separately, as a real setup would.

```{code-cell} ipython3
omega_sig = g.omega + scale * omega0
lam = wlfreq(omega_sig)
order = np.argsort(lam)
lam = lam[order]
meas = croak.preprocess.omega_to_lambda_density(trace[order], lam)
meas /= meas.max()
keep = (lam > 200e-9) & (lam < 360e-9)
lam, meas = lam[keep], meas[keep]
meas = meas + 1e-3 * np.random.default_rng(1).standard_normal(meas.shape)

omega_fund = g.omega + omega0
lam_spec = wlfreq(omega_fund)
order = np.argsort(lam_spec)
lam_spec = lam_spec[order]
Ilam_spec = croak.preprocess.omega_to_lambda_density(np.abs(ew[order]) ** 2, lam_spec)
inside = (lam_spec > 500e-9) & (lam_spec < 1400e-9)
lam_spec, Ilam_spec = lam_spec[inside], Ilam_spec[inside] / Ilam_spec[inside].max()
```

## Check the marginal before retrieving

The delay-integrated trace (the frequency marginal) should sit near
$3\bar\omega_S$ and be broader than the fundamental spectrum. For THG both are
*soft* anchors: the centroid holds exactly only when the instantaneous frequency
is odd about a symmetric intensity, and the transform-limited width is a
reference, not a bound. {func}`~croak.marginal_checks.predicted_marginal`
computes them from the fundamental spectrum alone:

```{code-cell} ipython3
omega_t = wlfreq(lam)
marg_omega = np.clip(meas.sum(axis=1), 0, None) * lam**2      # λ-density -> ω-density

omega_s = wlfreq(lam_spec)
s_omega = Ilam_spec * lam_spec**2
o = np.argsort(omega_s)
omega_u = np.linspace(omega_s.min(), omega_s.max(), 2048)
pred = mc.predicted_marginal(omega_u, np.interp(omega_u, omega_s[o], s_omega[o]), "thg")

c_meas = mc.centroid(omega_t, marg_omega)
w_meas = mc.rms_width(omega_t, marg_omega)
print(f"centroid: measured {wlfreq(c_meas) / 1e-9:.1f} nm, "
      f"predicted {wlfreq(pred.centroid) / 1e-9:.1f} nm (soft: {not pred.centroid_is_hard})")
print(f"width   : measured {w_meas / 1e12:.0f} rad/ps  >  σ_S = "
      f"{pred.width_bracket[0] / 1e12:.0f} rad/ps (rigorous lower bound)")

fig, ax = plt.subplots(figsize=(5.5, 3.5))
o_t = np.argsort(omega_t)
ax.plot(omega_t[o_t] / (2e15 * np.pi), marg_omega[o_t] / marg_omega.max(),
        label="measured marginal")
ax.plot(pred.omega / (2e15 * np.pi), pred.marginal, "--", label="TL prediction")
ax.axvline(pred.centroid / (2e15 * np.pi), color="C1", lw=1, ls=":")
ax.set_xlim(0.95, 1.3)
ax.set_xlabel("signal frequency (PHz)")
ax.set_ylabel("normalised marginal")
ax.legend()
fig.tight_layout()
```

The measured centroid sits a few nanometres off the prediction. That is the
TOD at work, not a calibration error: this pulse's instantaneous frequency is
not odd about its peak, so the soft anchor moves. The width comfortably clears
the rigorous lower bound. A marginal *narrower* than $\sigma_S$, or one
displaced by tens of nanometres, would point at the spectrometer instead.

## Size the band from the trace

`load_and_clean` builds the retrieval grid from the **fundamental** band
`lam_min`/`lam_max`, and places the trace on a window of the *same frequency
width* around $3\omega_0$. A band that merely covers the fundamental spectrum is
far too narrow for the THG trace:

```{code-cell} ipython3
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")      # the taper warning is the point here
    naive = croak.load_and_clean(meas, lam, delays, "thg",
                                 lam_min=700e-9, lam_max=920e-9, input_unit="delay")
print(f"700–920 nm band: the edge taper removes {naive.taper_loss:.0%} of the trace")
```

Instead, read the trace's edges off its marginal and map them back, as described
in [Grids and filtering](../howto/grids_and_filtering.md#sizing-the-band-for-a-harmonic-trace).
The trace window is centred on $n\omega_0$ with the band's full width, so
$\omega_0 = (\Omega_a+\Omega_b)/2n$ and the band is $\omega_0 \mp (\Omega_b-\Omega_a)/2$.
{func}`~croak.grid.fundamental_band` does that mapping:

```{code-cell} ipython3
marg = np.clip(meas.sum(axis=1), 0, None)
edge = np.where(marg > 3e-3 * marg.max())[0]       # above this trace's noise floor
lam_trace_min, lam_trace_max = lam[edge[0]], lam[edge[-1]]
lam_min, lam_max = croak.fundamental_band(lam_trace_min, lam_trace_max, scale)
print(f"trace spans {lam_trace_min / 1e-9:.0f}–{lam_trace_max / 1e-9:.0f} nm "
      f"-> fundamental band {lam_min / 1e-9:.0f}–{lam_max / 1e-9:.0f} nm "
      f"(not {3 * lam_trace_min / 1e-9:.0f}–{3 * lam_trace_max / 1e-9:.0f} nm)")

td = croak.load_and_clean(meas, lam, delays, "thg",
                          lam_min=lam_min, lam_max=lam_max, input_unit="delay",
                          lam_spec=lam_spec, Ilam_spec=Ilam_spec)
print(f"taper loss {td.taper_loss:.2%}; grid {td.grid.n} points, "
      f"dt = {td.grid.dt / 1e-15:.2f} fs")
```

On a noise-free trace you would cut at about 1e-4 of the peak. Here the noise
floor sets the threshold instead.

## Retrieve

The production default, `warm-lbfgs`, with the independent spectrum binding the
settled regularisation:

```{code-cell} ipython3
result = croak.retrieve_from_tracedata(td, rng=np.random.default_rng(0))
pr = croak.process_result(result, measured=td.trace)
print(f"FROG error R = {result.error:.2e}")
print(f"FWHM {pr.fwhm_retr / 1e-15:.2f} fs (true {true_fwhm / 1e-15:.2f} fs)")
print(f"GDD {pr.gdd_fs2:.0f} fs² (true 40),  TOD {pr.tod_fs3:.0f} fs³ (true 600)")
```

The error sits at the 0.1 % noise floor, and the duration, the GDD *and its
sign*, and the TOD all come back, because the TOD breaks the chirp-sign
symmetry. The overview plot shows every panel on the correct axes: the trace at
the third harmonic, and the spectrum and phase at the fundamental.

```{code-cell} ipython3
fig = croak.plot_retrieval(result, measured=td.trace, Iomega_meas=td.Iomega,
                           lam_min=td.lam_min, lam_max=td.lam_max,
                           truth=truth, processed=pr)
fig.set_size_inches(11, 8)
fig
```

## Takeaways

- Pass `"thg"` anywhere an interaction is accepted; every solver supports it in
  a thin medium. Dispersive propagation and geometric smearing do not apply.
- Size `lam_min`/`lam_max` from the *trace's* frequency span, not the
  fundamental spectrum. `td.taper_loss` tells you if you got it wrong.
- The marginal anchors are soft: expect the centroid near $3\bar\omega_S$ and
  the width above $\sigma_S$.
- Be sceptical of the chirp sign on a near-Gaussian, GDD-dominated pulse.
