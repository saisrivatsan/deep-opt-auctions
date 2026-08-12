"""Tullock contest success functions from Definition 2.1 of the paper."""

from __future__ import annotations

import torch  # type: ignore[import]
from torch import Tensor  # type: ignore[import]


def tullock_contest(bids: Tensor, exponent: float | Tensor) -> Tensor:
    """Return Tullock contest shares for a batch of bid profiles.

    This implements the *undiscounted* CSF ``tau`` from Definition 2.1, not
    the discounted allocation ``a^n`` from Definition 2.2. For a positive
    discrimination exponent ``k``, identity ``i`` receives the contest share
    contest share

        tau_i(b; k) = b_i**k / sum_j b_j**k.

    The paper defines the ``k = 0`` mechanism as the uniform lottery over
    every reported identity, including identities that bid zero. An all-zero
    profile is also assigned the uniform lottery for every nonnegative
    exponent. Consequently, when ``k > 0`` and every rival bid is zero, an
    identity's share jumps from ``1 / n`` at its own zero bid to ``1`` at any
    positive own bid. This endpoint convention is part of the paper's
    definition. It does not alter an envelope integral, but payment quadrature
    must not give that single endpoint positive mass.

    The positive-exponent calculation is performed as a softmax of centered
    log bids. This avoids forming ``bids**exponent`` directly and remains
    stable for large exponents. Exact zero bids are masked out without taking
    ``log(0)``, which also keeps gradients finite at the boundary.

    Args:
        bids: Floating-point tensor with shape ``(..., num_identities)``.
            Bids must be finite and nonnegative, and the last dimension must
            be nonempty.
        exponent: A finite, nonnegative scalar. It may be a scalar tensor so
            that positive exponents can be learned by gradient descent.

    Returns:
        A tensor with the same shape, dtype, and device as ``bids``. Shares
        are nonnegative and sum to one along the final dimension.

    Raises:
        TypeError: If ``bids`` is not a floating-point tensor.
        ValueError: If an input is malformed or outside the function domain.
    """
    if not isinstance(bids, Tensor):
        raise TypeError("bids must be a torch.Tensor")
    if not bids.is_floating_point():
        raise TypeError("bids must use a floating-point dtype")
    if bids.ndim == 0 or bids.shape[-1] == 0:
        raise ValueError("bids must have a nonempty identity dimension")
    if not bool(torch.isfinite(bids).all()):
        raise ValueError("bids must be finite")
    if bool((bids < 0).any()):
        raise ValueError("bids must be nonnegative")

    k = torch.as_tensor(exponent, dtype=bids.dtype, device=bids.device)
    if k.numel() != 1:
        raise ValueError("exponent must be a scalar")
    if not bool(torch.isfinite(k)):
        raise ValueError("exponent must be finite")
    if bool(k < 0):
        raise ValueError("exponent must be nonnegative")

    num_identities = bids.shape[-1]
    uniform = torch.full_like(bids, 1.0 / num_identities)

    # Use a positive exponent in the unselected branch when k == 0. This
    # prevents the indeterminate product 0 * (-inf) while preserving the exact
    # uniform-lottery definition and its zero derivative at k == 0.
    positive_k = torch.where(k == 0, torch.ones_like(k), k)

    positive_bid = bids > 0
    all_zero = ~positive_bid.any(dim=-1, keepdim=True)
    active_bid = positive_bid | all_zero

    # Clamp only to make log safe. Exact zero bidders are masked below, while
    # an all-zero row keeps identical finite logits and therefore stays uniform.
    log_bids = torch.log(bids.clamp_min(torch.finfo(bids.dtype).tiny))
    log_bids = log_bids - log_bids.max(dim=-1, keepdim=True).values
    logits = positive_k * log_bids
    logits = logits.masked_fill(~active_bid, -torch.inf)
    positive_shares = torch.softmax(logits, dim=-1)

    return torch.where(k == 0, uniform, positive_shares)
