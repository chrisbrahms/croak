# FROG trace marginals as consistency checks

The **frequency marginal** of a FROG-type trace is its integral over delay,

```{math}
M(\omega) \;=\; \int T(\omega,\tau)\,\mathrm{d}\tau ,
```

a one-dimensional function of optical frequency. It is computed from a measured
trace as `M_meas = trace.sum(axis=1)` (delay is axis 1 in croak's `(Nω, Nτ)`
convention). From an **independently measured fundamental spectrum**
$S(\omega)=|\tilde E(\omega)|^2$ we can predict $M(\omega)$ — or rigorously bound
it — and compare, *before* running a retrieval. A mismatch is a cheap
quality-control (QC) signal: it flags calibration errors, spectral-response
problems, scatter and detector nonlinearity that a retrieval would otherwise
silently absorb.

These checks are implemented in {mod}`croak.marginal_checks` and surfaced in the
GUI's **Marginal check** stage (see the [GUI guide](../howto/gui.md)); the
how-to recipe is [Marginal consistency checks](../howto/marginal_checks.md).

## The general result

Every FROG-type signal factorises into an *undelayed* part $P$ and a *delayed
gate* $g$,

```{math}
E_\mathrm{sig}(t,\tau) = P(t)\,g(t-\tau),
\qquad T(\omega,\tau)=\bigl|\widehat{E_\mathrm{sig}}(\omega,\tau)\bigr|^2 .
```

Integrating over delay collapses the delay-dependent phase to a delta function,
leaving a **convolution of spectral intensities**:

```{math}
M(\omega) \;\propto\; \bigl(|\tilde P|^2 * |\tilde g|^2\bigr)(\omega),
```

where $\tilde P,\tilde g$ are the Fourier transforms of $P,g$. Everything below
follows from this one identity.

Two consequences are **exact for any nonnegative factors** (no assumptions on the
pulse), because intensity-weighted moments add under convolution:

- **Centroid:** $\bar\omega[M] = \bar\omega[\,|\tilde P|^2\,] + \bar\omega[\,|\tilde g|^2\,]$
- **Variance:** $\sigma^2[M] = \sigma^2[\,|\tilde P|^2\,] + \sigma^2[\,|\tilde g|^2\,]$

with $\bar\omega[f]=\frac{\int\omega f}{\int f}$ and
$\sigma^2[f]=\frac{\int(\omega-\bar\omega)^2 f}{\int f}$
({func}`croak.marginal_checks.centroid` and
{func}`croak.marginal_checks.rms_width`).

**Phase-independence criterion.** A factor is phase-independent iff it is built
from $E$ or $E^\ast$ alone:

| factor | expression | phase-dependent? | centroid | even? |
|---|---|---|---|---|
| $\lvert\tilde E\rvert^2$ | $S(\omega)$ | no | $\bar\omega_S$ | no |
| $\lvert\widehat{E^\ast}\rvert^2$ | $S(-\omega)$ | no | $-\bar\omega_S$ | no |
| $\lvert\widehat{\lvert E\rvert^2}\rvert^2$ | intensity spectrum | **yes** | $0$ (exact) | **yes** |
| $\lvert\widehat{E^2}\rvert^2$ | SHG spectrum | **yes** | phase-dependent | no |

The marginal is fully phase-independent only when *both* factors come from the
top two rows. Among the common geometries that is **SHG only**.

| process | $P$ | $g$ | marginal $M\propto$ | phase-indep.? | centred at |
|---|---|---|---|---|---|
| SHG | $E$ | $E$ | $S * S$ | **yes** | $2\bar\omega_S$ |
| PG  | $E$ | $\lvert E\rvert^2$ | $S * \lvert\widehat{\lvert E\rvert^2}\rvert^2$ | no | $\bar\omega_S$ |
| SD  | $E^2$ | $E^\ast$ | $\lvert\widehat{E^2}\rvert^2 * S(-\omega)$ | no | $\approx\bar\omega_S$ |
| THG | $E^2$ | $E$ | $\lvert\widehat{E^2}\rvert^2 * S$ | no | $\approx 3\bar\omega_S$ |
| TG  | $E$ | $\lvert E\rvert^2$ | same as PG | no | $\bar\omega_S$ |

```{admonition} TG-FROG ≡ PG
:class: tip
The transient-grating geometry has the $|E|^2$ gate, so its marginal is
identical to PG regardless of which beam carries the delay (convolution
commutes). {func}`croak.marginal_checks.predicted_marginal` accepts
`interaction="tg"` and treats it as `"pg"`.
```

## SHG — the clean case

```{math}
M_\mathrm{SHG}(\omega) = (S * S)(\omega).
```

Both factors are the measured spectrum, so the marginal is the **autoconvolution
of the fundamental spectrum** ({func}`croak.marginal_checks.spectral_autoconvolution`)
— completely phase-independent and computable from $S$ alone. This is the
gold-standard check: any discrepancy is unambiguously a *trace* problem
(calibration, spectral response, detector nonlinearity, scatter), never a
property of the pulse. All three of these are **hard** checks:

- **Full predicted marginal:** $S*S$, compared shape-for-shape.
- **Centroid:** $\bar\omega[M] = 2\,\bar\omega_S$.
- **RMS width:** $\sigma[M] = \sqrt{2}\,\sigma_S$.

Place the prediction on the second-harmonic frequency axis and compare to the
measured marginal after consistent wavelength calibration.

## PG (and TG) — three checks

```{math}
M_\mathrm{PG}(\omega) = \bigl(S * \lvert\widehat{\lvert E\rvert^2}\rvert^2\bigr)(\omega).
```

The gate factor $\lvert\widehat{\lvert E\rvert^2}\rvert^2$ is the modulus-squared
FT of the **temporal intensity** $|E(t)|^2$. It carries the spectral phase
through $e^{i[\phi(\omega'+\omega)-\phi(\omega')]}$, so you cannot get it from $S$
alone. Three things are nonetheless usable.

### Transform-limited (TL) predicted marginal

Assume flat phase: $\tilde E_\mathrm{TL}=\sqrt{S}$. Then $|E_\mathrm{TL}(t)|^2$
and the gate factor are computable, so the whole marginal is computable from $S$.
Compare its **shape** to the measured marginal. A mismatch conflates "bad trace"
with "pulse is not TL", so this is a *soft* check, most useful alongside the
centroid and width anchors below.

### Centroid — exact, phase-independent

The gate factor is the modulus-squared FT of a **real** function, hence **even**
in $\omega$, hence has centroid exactly $0$ for any pulse. Therefore

```{math}
\bar\omega[M_\mathrm{PG}] = \bar\omega_S \quad\text{(exact, any phase).}
```

PG/TG sit at the fundamental (not doubled). This is a genuine **hard** check on
the frequency-axis calibration of the trace, independent of the TL assumption.

### Width — a two-sided bracket

Variances add: $\sigma^2[M_\mathrm{PG}] = \sigma_S^2 + \sigma^2[\,|\widehat{|E|^2}|^2\,]$.

- **Lower bound (rigorous):** $\sigma[M_\mathrm{PG}] > \sigma_S$. The gate term is
  strictly positive (it vanishes only for a CW field). A measured marginal
  *narrower* than the bare spectrum is unphysical — typically spectral clipping
  or detector roll-off.
- **Upper reference (soft):** the TL pulse has the most peaked $|E(t)|^2$, hence
  the broadest gate factor and the broadest marginal — *for smooth phase*.
  Pathological phase can beat $|E(t)|^2$ faster and exceed it, so this is
  typical-case, not a theorem. A measured marginal *broader* than the TL
  prediction flags additive broadband artefacts (scatter, background, stray
  light, detector nonlinearity).

So the expected width sits in the bracket
$\bigl[\sigma_S,\ \sigma_\mathrm{TL}\bigr]$
({attr}`~croak.marginal_checks.MarginalPrediction.width_bracket`), the interior
consistent with some unknown amount of chirp.

## SD — partly similar, with weaker anchors

```{math}
M_\mathrm{SD}(\omega) = \bigl(\lvert\widehat{E^2}\rvert^2 * S(-\omega)\bigr)(\omega).
```

The signal is $E_\mathrm{sig}=E(t)^2\,E^\ast(t-\tau)$, so the undelayed factor is
$E^2$ and its spectral intensity $\lvert\widehat{E^2}\rvert^2$ is the **SHG
spectrum** — phase-dependent. The delayed factor contributes only the reflected
fundamental spectrum $S(-\omega)$, which is phase-independent. **SD does not give
a phase-independent marginal**: the phase enters through the SHG-spectrum factor,
exactly as it would in an SHG measurement. Only SHG-FROG has the fully clean
marginal.

How the three PG checks carry over:

- **TL predicted marginal — fully analogous.** Set $\tilde E_\mathrm{TL}=\sqrt S$,
  form $E_\mathrm{TL}^2$, take $|\widehat{E_\mathrm{TL}^2}|^2$, convolve with
  $S(-\omega)$. Same soft-check status as PG.
- **Centroid — weaker.**
  $\bar\omega[M_\mathrm{SD}] = \bar\omega[\,|\widehat{E^2}|^2\,] - \bar\omega_S$.
  The SHG-spectrum centroid is *not* generally $2\bar\omega_S$. Writing $E(t)=|E|e^{i\phi(t)}$, the SHG-spectrum centroid (about $2\omega_0$)
  is $2\langle\phi'\rangle_{|E|^4}$, twice the instantaneous frequency averaged
  with weight $|E|^4$, whereas $\bar\omega_S-\omega_0=\langle\phi'\rangle_{|E|^2}$.
  The two averages agree when $\phi'$ is constant. They also agree when $\phi'$
  is **odd about a symmetric intensity profile**, which covers a flat-phase pulse
  with a symmetric spectrum and any linear chirp of one. A symmetric intensity
  alone is *not* enough: a symmetric pulse with a cubic temporal phase moves the
  centroid by ~0.7 $\sigma_S$. Even a transform-limited pulse with an asymmetric
  spectrum is off by a few per cent of $\sigma_S$.
  So treat "centred at the fundamental" as a TL/symmetric anchor, not a hard check
  ({attr}`~croak.marginal_checks.MarginalPrediction.centroid_is_hard` is `False`).
- **Width — messier.** Only the lower bound $\sigma[M_\mathrm{SD}] > \sigma_S$ is
  rigorous; the SHG-spectrum width depends on phase non-monotonically (for a
  Gaussian spectrum with pure GDD it is essentially invariant), so the clean
  PG-style upper bracket is lost — the prediction reports
  `width_bracket = (σ_S, None)`.
- **Bonus check (no TL assumption):** if an **independently measured SHG
  spectrum** is available, plug it in for $|\widehat{E^2}|^2$:
  $M_\mathrm{SD} = (\text{measured SHG spec}) * S(-\omega)$ is then a fully
  phase-independent prediction, recovering an SHG-grade check for the SD trace.

## THG — SD's anchors on the tripled axis

```{math}
M_\mathrm{THG}(\omega) = \bigl(\lvert\widehat{E^2}\rvert^2 * S\bigr)(\omega).
```

The signal is $E_\mathrm{sig}=E(t)^2\,E(t-\tau)$. The undelayed factor is again
$E^2$, whose spectral intensity is the phase-dependent SHG spectrum. The gate is
the field itself, unconjugated, so it contributes $S(\omega)$ rather than SD's
mirrored $S(-\omega)$. The marginal therefore sits near the **third harmonic**,
and the checks carry over from SD:

- **TL predicted marginal** — $|\widehat{E_\mathrm{TL}^2}|^2 * S$. For a Gaussian
  spectrum it has RMS width exactly $\sqrt3\,\sigma_S$. It is a soft check.
  {func}`~croak.marginal_checks.predicted_marginal` places this shape on the
  analytic $3\bar\omega_S$, as it places SD's on $\bar\omega_S$. For a
  transform-limited pulse with an *asymmetric* spectrum, the true TL marginal
  centroid differs from that by a few per cent of $\sigma_S$.
- **Centroid — soft.**
  $\bar\omega[M_\mathrm{THG}] = \bar\omega[\,|\widehat{E^2}|^2\,] + \bar\omega_S$,
  which equals $3\bar\omega_S$ under exactly SD's condition: the instantaneous
  frequency is odd about a symmetric intensity profile. That includes a linearly
  chirped Gaussian but not a symmetric pulse with a cubic temporal phase.
- **Width — lower bound only.** $\sigma[M_\mathrm{THG}] > \sigma_S$ holds for any
  pulse. As for SD, `width_bracket = (σ_S, None)`.
- **Bonus check.** An independently measured SHG spectrum plugged in for
  $|\widehat{E^2}|^2$ makes the THG prediction phase-independent too.

Because the THG signal is in the UV, the spectrometer's response is the usual
culprit when these anchors fail. Check the centroid against $3\bar\omega_S$
before trusting a retrieved chirp.

## What to compute, by process

| check | SHG | PG / TG | SD | THG |
|---|---|---|---|---|
| predicted marginal from $S$ alone | exact | TL only | TL only (or via measured SHG spec) | TL only (or via measured SHG spec) |
| centroid | $2\bar\omega_S$ (hard) | $\bar\omega_S$ (hard) | $\approx\bar\omega_S$ at TL/symmetric (soft) | $\approx 3\bar\omega_S$ at TL/symmetric (soft) |
| width | $\sqrt2\,\sigma_S$ (hard) | bracket $[\sigma_S,\ \sigma_\mathrm{TL}]$ | $>\sigma_S$; TL estimate (weak) | $>\sigma_S$; TL estimate (weak) |
| nature of a mismatch | trace error only | trace error or non-TL phase | trace error or phase | trace error or phase |

## Practical points

- **Normalise** the predicted and measured marginals (unit peak or unit area)
  before comparing shape; the absolute scale is set by the conversion efficiency
  and detector gain. {func}`~croak.marginal_checks.predicted_marginal` returns a
  peak-normalised marginal.
- Compare on a **common, calibrated frequency axis**: the SH axis for SHG, the
  TH axis for THG, the fundamental axis for PG/TG/SD. A centroid mismatch on that axis *is* the
  calibration check.
- The width and centroid checks are robust to additive noise; the full-shape
  overlay is the most sensitive to scatter and detector nonlinearity but also the
  most affected by the TL assumption for PG/SD/THG. Restrict the moments and the
  normalisation to a **spectral trust region** (in the GUI, 190–1000 nm for a
  measured trace, or the full loaded range — trace and independent spectrum — for
  the noise-free simulated/synthetic traces) so out-of-band detector noise cannot
  dominate.
- For the rigorous SHG and PG-centroid checks, report the numeric discrepancy
  (measured vs predicted) as a QC metric alongside the retrieval error.

## Centroid-matching efficiency correction

A TG/PG detector's third-order generation/collection efficiency tilts the
marginal by a power law $(\lambda/\mu\mathrm{m})^n$
({func}`croak.preprocess.third_order_scale`). The marginal centroid is a
**hard** anchor for PG/TG and SHG, but only a soft TL/symmetric anchor for SD
and THG, so on those geometries an automatic exponent also absorbs any
phase-induced centroid shift (such as the ~2 nm TOD shift in the
[THG tutorial](../tutorials/06_thg_frog.md)) as a spurious efficiency tilt.
{func}`croak.marginal_checks.auto_third_order_exponent` picks the exponent $n$
whose corrected-marginal centroid lands on the predicted centroid. The tilt
shifts the centroid monotonically in $n$, so the match is a unique root; it is
restricted to the trust region so amplified out-of-band noise cannot bias it.
This is the physically grounded replacement for fitting the marginal *shape* to
the spectrum.
