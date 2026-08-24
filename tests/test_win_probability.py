import pytest

from nba_game_projections.models import win_probability as wp


def test_normal_cdf_known_values():
    assert wp.normal_cdf(0.0) == pytest.approx(0.5)
    assert wp.normal_cdf(1.0) == pytest.approx(0.8413, abs=1e-4)
    assert wp.normal_cdf(-1.0) == pytest.approx(0.1587, abs=1e-4)


def test_worked_example():
    # residuals [-10.0, 10.0] have population std exactly 10.0.
    # z = 5.0 / 10.0 = 0.5 -> Phi(0.5) ~= 0.6914625
    result = wp.win_probability(5.0, [-10.0, 10.0])
    assert result == pytest.approx(0.6914625, abs=1e-4)


def test_zero_margin_is_coin_flip():
    residuals = [-10.0, -5.0, 0.0, 5.0, 10.0]
    assert wp.win_probability(0.0, residuals) == pytest.approx(0.5)


def test_monotonic_in_predicted_margin():
    residuals = [-10.0, -5.0, 0.0, 5.0, 10.0]
    margins = [-10.0, -1.0, 0.0, 1.0, 10.0]
    probs = [wp.win_probability(m, residuals) for m in margins]
    assert probs == sorted(probs)
    assert len(set(probs)) == len(probs)


def test_symmetry():
    residuals = [-12.0, -6.0, 0.0, 6.0, 12.0]
    for margin in (0.5, 3.0, 15.0):
        p_pos = wp.win_probability(margin, residuals)
        p_neg = wp.win_probability(-margin, residuals)
        assert p_pos + p_neg == pytest.approx(1.0, abs=1e-9)


def test_degenerate_zero_sigma():
    residuals = [0.0, 0.0, 0.0]
    assert wp.win_probability(5.0, residuals) == 1.0
    assert wp.win_probability(-5.0, residuals) == 0.0
    assert wp.win_probability(0.0, residuals) == 0.5
