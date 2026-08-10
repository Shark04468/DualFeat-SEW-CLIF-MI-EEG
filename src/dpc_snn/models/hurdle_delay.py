"""Differentiable online null-versus-positive-delay routing posterior.

The learned scores in this module are neural routing evidence.  They are not a
calibrated marginal likelihood and therefore must not be reported as a Bayes
factor.  Offline Bayes-factor estimation lives in ``analysis.evidence_space``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class HurdleDelayOutput:
    positive_probability: torch.Tensor
    route_probability: torch.Tensor
    null_probability: torch.Tensor
    route_log_odds: torch.Tensor


def hurdle_delay_posterior(
    score: torch.Tensor,
    route_odds_threshold: float = math.log(3.0),
    route_temperature: float = 0.5,
    force_zero_delay: bool = False,
    min_bayes_factor: float | None = None,
    route_logit: torch.Tensor | None = None,
) -> HurdleDelayOutput:
    """Separate online route acceptance from conditional transport delay.

    ``force_zero_delay`` is a matched transport control: it changes only the
    conditional delay distribution.  Route acceptance is computed from the
    same scores as the full model, so edge strength and sparsity are preserved.

    ``min_bayes_factor`` is accepted only for backward-compatible config
    loading and is converted to a log-odds threshold.  New code should pass
    ``route_odds_threshold`` directly.
    """

    if score.shape[-1] < 2:
        raise ValueError("Hurdle delay requires zero and at least one positive delay bin")
    positive_score = score[..., 1:]
    if route_logit is None:
        # Backward-compatible diagnostic path. Scientific training passes an
        # explicit Bernoulli route logit so the null is not part of lag softmax.
        zero_score = score[..., 0]
        route_log_odds = torch.logsumexp(positive_score, dim=-1)
        route_log_odds = route_log_odds - math.log(positive_score.shape[-1]) - zero_score
    else:
        if route_logit.shape != score.shape[:-1]:
            raise ValueError("route_logit must match score without its lag dimension")
        route_log_odds = route_logit
    threshold = float(route_odds_threshold)
    if min_bayes_factor is not None:
        threshold = math.log(max(float(min_bayes_factor), 1e-6))
    route = torch.sigmoid(
        (route_log_odds - threshold) / max(float(route_temperature), 1e-4)
    )
    if force_zero_delay:
        probability = torch.zeros_like(score)
        probability[..., 0] = 1.0
    elif route_logit is not None:
        # With an explicit hurdle, every entry is a transport base k whose
        # continuous delay is k + fraction. Base k=0 can therefore represent a
        # strictly positive sub-sample delay and is not the null hypothesis.
        probability = torch.softmax(score, dim=-1)
    else:
        conditional = torch.softmax(positive_score, dim=-1)
        probability = torch.cat((torch.zeros_like(conditional[..., :1]), conditional), dim=-1)
    return HurdleDelayOutput(
        positive_probability=probability,
        route_probability=route,
        null_probability=1.0 - route,
        route_log_odds=route_log_odds,
    )
