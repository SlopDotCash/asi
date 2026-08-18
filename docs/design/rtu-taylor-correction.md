# RTU Taylor-corrected approximate sensitivity

## Scope and source status

`RecurrentTraceActorCriticConfig.rtrl_taylor_correction` (in
`alberta_framework/core/recurrent_trace_actor_critic.py`) enables the
parameter-wise diagonal Taylor trace from Appendix C.2 of
[Farr et al., arXiv:2605.24709v2](https://arxiv.org/abs/2605.24709). It is
disabled by default. The result is a **Taylor-corrected approximate
sensitivity**, not exact RTRL under moving parameters.

The full rank-three `Omega` correction in Equation 15 has a second-order
Taylor residual. This implementation uses the parameter-wise diagonal
approximation in Equation 16. If several recurrent parameters move together,
omitted mixed-parameter Hessian terms can leave an `O(||Delta psi||)` residual,
and the correction is not guaranteed to reduce total staleness monotonically.

The paper cites the Apache-2.0
[Memorax repository at commit
`94dd21dba5ab6eee4a0281066a5f49643d052174`](https://github.com/noahfarr/memorax/tree/94dd21dba5ab6eee4a0281066a5f49643d052174).
That commit's
[`rtu.py`](https://github.com/noahfarr/memorax/blob/94dd21dba5ab6eee4a0281066a5f49643d052174/memorax/networks/sequence_models/rtu.py)
implements only the ordinary local-Jacobian/RTRL recurrence. The exact commit,
its public `development` branch, and the paper release repository contain no
Taylor-trace implementation. The implementation retained in ASI is therefore a
derivation from the paper's equations, not a port of undocumented reference
code.

## Paper recurrence and Alberta mapping

For recurrent parameters `psi`, the paper writes

```text
I_t     = partial h_t / partial psi
J_t     = partial h_t / partial h_(t-1)
omega_t = J_t omega_(t-1) + diag_psi(partial I_t / partial psi)
S_t     = J_t (S_(t-1) + omega_(t-1) * Delta psi_(t-1)) + I_t
```

where `Delta psi_(t-1) = psi_t - psi_(t-1)` and `*` is elementwise along the
parameter axis. Full `Omega` would have shape `|h| x p x p`; the paper's
parameter-wise diagonal `omega` has the same compressed shape as `S`.

The ordinary Alberta core retains only

```text
S_t = J_t S_(t-1) + I_t.
```

The optional path adds one `RTUSensitivities`-shaped `omega` tree per actor and
critic. It covers all recurrent parameter leaves: `nu_log`, `theta_log`,
`b_real`, and `b_imag`. It does not add a historical encoder sensitivity; the
encoder retains the paper's existing one-step approximation.

## RTU diagonal derivation

For either real RTU component, let `y = tanh(z)`,
`q_p = partial z / partial p`, and
`q2_p = partial^2 z / partial p^2`, holding the historical input and incoming
state fixed as required by the local immediate Jacobian `I_t`. Then

```text
partial I_p / partial p
  = tanh''(z) q_p^2 + tanh'(z) q2_p,
tanh'(z)  = 1 - y^2,
tanh''(z) = -2 y (1 - y^2).
```

With `e = exp(nu_log)`, `r = exp(-e)`,
`g = r cos(theta)`, `phi = r sin(theta)`, and
`n = sqrt(1 - r^2) + epsilon`, the needed coefficient derivatives are

```text
g_nu'    = -e g                    phi_nu'    = -e phi
g_nu''   = (e^2 - e) g             phi_nu''   = (e^2 - e) phi
n_nu'    = e r^2 / sqrt(1-r^2)
n_nu''   = e r^2(1-2e)/sqrt(1-r^2)
            - e^2 r^4/(1-r^2)^(3/2)

g_theta'  = -phi theta             phi_theta'  = g theta
g_theta'' = -g theta^2 - phi theta
phi_theta'' = -phi theta^2 + g theta.
```

The `b_real` and `b_imag` preactivation derivatives are `n x_f` in their
respective component and have zero second derivative with respect to the same
matrix element. The `tanh'' q^2` term remains, so both input-matrix leaves have
nonzero diagonal Hessian injections. Log-transform clamps and the
normalization floor use the same piecewise convention as ordinary Alberta
RTU sensitivities: derivatives are zero outside an active branch.

The analytic injection is checked both against a central difference of the
immediate sensitivity and against the full JAX Hessian diagonal for every leaf
in `tests/test_rtu_taylor_correction.py`.

## Online update order

At the beginning of an Alberta transition, `h_t`, `S_t`, and `omega_t` were
advanced with `psi_t`. The update order is:

1. Compute the current actor/critic gradients and eligibility traces.
2. Compute ObGD changes and update all network parameters.
3. Project `nu_log` and `theta_log` into the stable RTU transform interval.
4. Set `Delta psi_t` to the **actual projected recurrent-parameter difference**
   `psi_(t+1) - psi_t`, including every RTU leaf.
5. Encode the next stored observation with the updated network.
6. Advance `S_(t+1)` with `omega_t` and `Delta psi_t`; then advance
   `omega_(t+1)` using local derivatives evaluated at `psi_(t+1)`.

On an episode boundary, the next episode starts from zero `h`, `S`, and
`omega`, and the just-computed parameter delta is replaced by zero before its
first observation. This prevents a terminating update from correcting history
that belongs to the previous episode. `start` uses the same zero-history rule.

## Cost and evidence limits

Both ordinary compressed `S` and diagonal `omega` require
`O(H + H F)` storage for hidden width `H` and RTU input width `F`; enabling the
option adds one sensitivity-sized tree per network. When disabled, the two
optional state slots are `None` and add no array leaves to a JAX carry. The
implementation reuses the ordinary `rtu_step` for `S` and performs one
additional local derivative pass for `omega`, so its time remains
`O(H + H F)` with a larger constant.

Tests establish the local Hessian diagonal, bitwise reduction to ordinary RTRL
when `Delta psi` is zero, post-projection delta ordering, boundary isolation,
and first-order-versus-second-order error scaling against the formal Equation 8
fixed-trajectory target under controlled one-coordinate online motion. A
simultaneous-motion diagnostic separately establishes that the diagonal
approximation retains a first-order mixed-parameter residual even though the
full Hessian-vector Taylor correction has a second-order residual. The tests do
not recompute the paper's current-parameter full replay trajectory, establish
exact gradients for a generally moving multi-parameter recurrent network, or
support any benchmark-performance claim.
