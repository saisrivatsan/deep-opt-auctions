"""Tests for the Tullock contest success function."""

import pytest # type: ignore[import]
import torch  # type: ignore[import]

from wasteNet.mechanisms.tullock import tullock_contest


DTYPE = torch.float64


def test_matches_direct_formula() -> None:
    bids = torch.tensor([[1.0, 2.0, 3.0], [0.5, 0.25, 0.75]], dtype=DTYPE)
    exponent = 2.0

    expected = bids.pow(exponent) / bids.pow(exponent).sum(dim=-1, keepdim=True)

    torch.testing.assert_close(tullock_contest(bids, exponent), expected)


@pytest.mark.parametrize("exponent", [0.0, 0.5, 1.0, 10.0])
def test_shares_are_feasible(exponent: float) -> None:
    bids = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=DTYPE)

    shares = tullock_contest(bids, exponent)

    assert bool((shares >= 0).all())
    assert bool((shares <= 1).all())
    torch.testing.assert_close(shares.sum(dim=-1), torch.ones(2, dtype=DTYPE))


def test_zero_exponent_is_exact_uniform_lottery() -> None:
    bids = torch.tensor([[0.0, 1.0, 10.0], [5.0, 5.0, 0.0]], dtype=DTYPE)

    expected = torch.full_like(bids, 1.0 / 3.0)

    torch.testing.assert_close(tullock_contest(bids, 0.0), expected, rtol=0, atol=0)


@pytest.mark.parametrize("exponent", [0.5, 1.0, 100.0])
def test_all_zero_profile_is_uniform(exponent: float) -> None:
    bids = torch.zeros((2, 4), dtype=DTYPE)

    expected = torch.full_like(bids, 0.25)

    torch.testing.assert_close(
        tullock_contest(bids, exponent), expected, rtol=0, atol=0
    )


def test_zero_bid_has_zero_share_for_positive_exponent() -> None:
    bids = torch.tensor([0.0, 1.0, 2.0], dtype=DTYPE)

    shares = tullock_contest(bids, 2.0)

    assert shares[0] == 0
    torch.testing.assert_close(shares[1:], torch.tensor([0.2, 0.8], dtype=DTYPE))


def test_is_scale_invariant() -> None:
    bids = torch.tensor([[0.25, 1.5, 4.0]], dtype=DTYPE)

    shares = tullock_contest(bids, 3.5)
    scaled_shares = tullock_contest(1e100 * bids, 3.5)

    torch.testing.assert_close(shares, scaled_shares)


def test_own_share_is_monotone_in_own_bid() -> None:
    lower = torch.tensor([1.0, 2.0, 3.0], dtype=DTYPE)
    higher = lower.clone()
    higher[1] = 4.0

    assert tullock_contest(higher, 2.0)[1] > tullock_contest(lower, 2.0)[1]


def test_large_exponent_is_numerically_stable() -> None:
    bids = torch.tensor([[1e-200, 1.0, 1e200]], dtype=DTYPE)

    shares = tullock_contest(bids, 1e6)

    assert bool(torch.isfinite(shares).all())
    torch.testing.assert_close(shares, torch.tensor([[0.0, 0.0, 1.0]], dtype=DTYPE))


def test_positive_exponent_is_differentiable() -> None:
    bids = torch.tensor([[0.5, 1.0, 2.0]], dtype=DTYPE, requires_grad=True)
    exponent = torch.tensor(1.5, dtype=DTYPE, requires_grad=True)

    objective = tullock_contest(bids, exponent)[0, 2]
    objective.backward()

    assert bids.grad is not None
    assert exponent.grad is not None
    assert bool(torch.isfinite(bids.grad).all())
    assert bool(torch.isfinite(exponent.grad))


@pytest.mark.parametrize(
    ("bids", "exponent", "error", "message"),
    [
        (torch.tensor([1, 2]), 1.0, TypeError, "floating-point"),
        (torch.tensor([-1.0, 2.0]), 1.0, ValueError, "nonnegative"),
        (torch.tensor([1.0, torch.inf]), 1.0, ValueError, "finite"),
        (torch.tensor([1.0, 2.0]), -1.0, ValueError, "nonnegative"),
        (torch.tensor([1.0, 2.0]), torch.inf, ValueError, "finite"),
    ],
)
def test_rejects_invalid_inputs(
    bids: torch.Tensor,
    exponent: float | torch.Tensor,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        tullock_contest(bids, exponent)
