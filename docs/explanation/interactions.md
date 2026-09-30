# Interactions

An **interaction** defines the instantaneous nonlinear signal field that a FROG
geometry produces, together with its reverse-mode (Wirtinger) adjoint. They live
in {mod}`croak.interactions` and are the only geometry-specific piece of the
forward model.

Each interaction takes a *test* field $E$ (undelayed) and a *gate* field $G$
(delayed by $\tau$), both in the time domain, and returns the signal $s(t)$.

## The four geometries

| Class | `name` | Signal $s$ | $\omega_0$ scale | fixes direction of time? |
|-------|--------|-----------|------------------|--------------------------|
| {class}`~croak.interactions.SHG` | `"shg"` | $E\,G$ | 2.0 | no |
| {class}`~croak.interactions.THG` | `"thg"` | $E^2\,G$ | 3.0 | yes (see below) |
| {class}`~croak.interactions.SD` | `"sd"` | $E^2\,G^*$ | 1.0 | yes |
| {class}`~croak.interactions.PG` | `"pg"` | $E\,\lvert G\rvert^2$ | 1.0 | yes |

`omega0_scale` records where the signal sits relative to the pulse carrier — the
harmonic order of the signal: the SHG signal is generated near $2\omega_0$, the
THG signal near $3\omega_0$, while the $\chi^{(3)}$ SD and PG signals stay at
the fundamental $\omega_0$. This single number has two consequences:

- it sets the wavelength axis of a harmonic FROG trace (handled in
  [Grids and filtering](../howto/grids_and_filtering.md));
- it determines which geometries support **dispersive propagation**. The slab
  propagation constant $\beta(\omega)$ is built at the fundamental carrier, so it
  is only physically consistent for signals that stay there. **Dispersive
  propagation is therefore supported for PG and SD, but not SHG or THG** — croak
  raises a `ValueError` if you request a dispersive harmonic model.

The last column is {attr}`~croak.interactions.Interaction.time_reversal_ambiguous`:
whether the trace is invariant under time reversal $E(t)\to E^*(-t)$, which maps
any trace to its delay mirror image $I(\omega,-\tau)$. Only SHG, whose signal is
symmetric in its two arms, has a delay-symmetric trace for every pulse and so
cannot tell a pulse from its time reverse; see
{func}`croak.processing.resolve_time_direction`.

```{admonition} TG-FROG uses the PG kernel
:class: tip
Transient-grating FROG — the geometry demonstrated in the
[companion paper](validation.md) — has the third-order signal
$E_1 E_2^* E_3$ with three replicas of the pulse. For a single unknown pulse this
reduces to $E\,\lvert G\rvert^2$, i.e. the **PG** kernel. Use
`interaction="pg"` for TG-FROG data.
```

## THG-FROG

Third-harmonic-generation FROG mixes the undelayed pulse twice with a delayed
replica, $E_\mathrm{sig}(t,\tau) = E(t)^2\,E(t-\tau)$, and spectrally resolves the
signal near $3\omega_0$. In practice the signal comes from the $\chi^{(3)}$
response of a thin plate or from a surface or interface, where the third harmonic
is free of phase-matching limits. That makes THG-FROG attractive for few-cycle
pulses, whose bandwidth no practical SHG crystal can phase-match: the
signal of a Ti:sapphire pulse lands near 267 nm and needs only a UV spectrometer.
The same physics sets its costs. The third harmonic is weak, the signal band is
roughly $\sqrt3$ times broader than the fundamental, and a UV detector's spectral
response must be calibrated (see [Marginals](marginals.md) for the
consistency checks).

Unlike SHG, THG treats its two arms differently ($E^2$ against $G$), so a
general THG trace is **not** symmetric in delay and the direction of time is
fixed. There is one important special case. For a pulse whose spectrum is
Gaussian and whose phase is purely quadratic (GDD only), $E(t)^2E(t-\tau)$ is a
Gaussian times a linear chirp, both centred on $t=\tau/3$. Its spectral
intensity then depends on $\tau$ only through $\tau^2$. **The THG trace of a
linearly chirped Gaussian is exactly symmetric in delay, so it cannot tell
$+$GDD from $-$GDD.** For pulses close to that case, such as a near-Gaussian
spectrum with mostly GDD, the chirp sign is only weakly encoded, typically at
the per-cent level of the trace. Any third-order phase or temporal-intensity
asymmetry breaks the symmetry and the sign becomes measurable again. Two
practical consequences follow:

- start the retrieval from a guess with a *small* chirp rather than an exactly
  flat phase. A flat-phase guess sits on the saddle between the two mirror
  solutions and converges slowly;
- when the sign matters and the pulse is nearly a chirped Gaussian, confirm it
  independently, for example by adding a known amount of glass. SD- or PG-FROG
  see the sign directly.

THG has no three-arm form in croak, so [geometric smearing](forward_model.md#geometric-smearing)
and dispersive propagation are unavailable for it (both raise a clear
`ValueError`). SHG-only [marginal correction](marginals.md) does not apply to it
either.

## Resolving an interaction

Anywhere an interaction is expected you may pass its name (case-insensitive) or an
instance; {func}`~croak.interactions.get_interaction` resolves both, and
{data}`~croak.interactions.INTERACTIONS` is the name→instance registry.

```python
from croak.interactions import get_interaction, PG
get_interaction("PG")    # -> PG()
get_interaction(PG())    # -> the same instance, unchanged
```

## The adjoint convention

croak computes exact gradients by hand rather than by automatic differentiation
(see [Gradients](gradients.md)). Each interaction therefore provides not only
`signal(test, gate)` but also `signal_adjoint(test, gate, s_bar)`, which
back-propagates a cotangent through the nonlinearity.

The convention is **Wirtinger**: the cotangent of a complex variable $v$ is

```{math}
\bar v \;\equiv\; \frac{\partial L}{\partial v^*},
```

which is the quantity gradient descent on a real loss $L$ actually needs. Given
the output cotangent $\bar s$, `signal_adjoint` returns $(\bar E, \bar G)$.

For example, polarisation gating $s = E\lvert G\rvert^2$ has

```{math}
\bar E = \lvert G\rvert^2\,\bar s,
\qquad
\bar G = 2\,G\,\operatorname{Re}\!\bigl(E^*\,\bar s\bigr),
```

which is exactly what {meth}`PG.signal_adjoint <croak.interactions.PG.signal_adjoint>`
implements. The holomorphic SHG and THG kernels and the mixed-holomorphy SD
kernel have similarly compact adjoints; see the [Gradients](gradients.md) page for the full
set and the derivation.

## Extending to a new geometry

A new geometry is a small subclass. As an illustration, a hypothetical
fourth-harmonic FROG with $s = E^3 G$ (holomorphic, so each cotangent is the
conjugated derivative times $\bar s$):

```python
import numpy as np
from croak.interactions import Interaction

class FHG(Interaction):           # fourth-harmonic generation, s = E^3 G
    name = "fhg"
    omega0_scale = 4.0            # signal near 4*omega0

    def signal(self, test, gate):
        return test**3 * gate

    def signal_adjoint(self, test, gate, s_bar):
        test_bar = 3.0 * np.conj(test**2 * gate) * s_bar
        gate_bar = np.conj(test**3) * s_bar
        return test_bar, gate_bar
```

Built-in geometries are wired up in four places:

1. the class and its entry in {data}`~croak.interactions.INTERACTIONS`, which
   the NumPy forward model and solvers ({class}`~croak.COPRA`,
   {class}`~croak.LBFGS`) use unchanged;
2. the JAX twins in `croak/forward_jax.py`. These are the name-keyed
   `_JAX_INTERACTIONS` (the signal, used by every autodiff solver) and
   `_JAX_INTERACTION_ADJOINTS` (the hand adjoint, used by `lbfgs-hand` and
   `copra-jax`). Without them the JAX solvers raise a `KeyError`;
3. a frequency-marginal model in {func}`croak.marginal_checks.predicted_marginal`,
   which raises for a geometry it does not know;
4. the tests. The finite-difference adjoint check in `tests/test_interactions.py`
   and the AD-versus-hand-adjoint check in `tests/test_ad_vs_analytic.py` are
   what catch a wrong `signal_adjoint`.

Comparing the JAX autodiff gradient with the hand-derived `signal_adjoint` is the
quickest way to validate a new adjoint.

To support [geometric smearing](forward_model.md#geometric-smearing) as well, add
{meth}`~croak.interactions.Interaction.smeared_shifts` and
{meth}`~croak.interactions.Interaction.smeared_signal`: the same signal written as a
product of *three* separately shifted replicas, because the factors that coincide
in the 1-D model acquire different arrival times across the focal spot. PG and SD
implement them; a geometry without a three-arm form (SHG, THG) simply does not, and
the forward model raises if a kernel is supplied.

## API

See {mod}`croak.interactions` in the [API reference](../reference/api/index.md).
