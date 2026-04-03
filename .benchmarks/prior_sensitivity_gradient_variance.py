#!/usr/bin/env python
"""Benchmark pathwise and REINFORCE gradient variance for prior sensitivity.

This script compares three estimators for gradients of posterior functionals
with respect to prior hyperparameters in a Bayesian linear model:

1. Pathwise gradients through ``blackjax.reparameterized_slice``
2. Naive REINFORCE using the unnormalized joint score without a baseline
3. Corrected REINFORCE using the covariance form with a sample-mean baseline

The script is intentionally local-only and is not wired into the repo's test or
benchmark suites.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln
import numpy as np

import blackjax

jax.config.update("jax_enable_x64", True)


class Dataset(NamedTuple):
    x: jax.Array
    y: jax.Array
    init_position: jax.Array


class ReplicateAux(NamedTuple):
    mean_functionals: jax.Array
    mean_functional_score: jax.Array
    mean_score: jax.Array


FUNCTIONAL_NAMES = ("mean_w1", "mean_cos_alpha")
ESTIMATOR_NAMES = (
    "pathwise",
    "reinforce_raw_joint_score",
    "reinforce_centered_covariance",
)
REFERENCE_PARAMETER_NAMES = ("a0", "b0", "lambda")
DEFAULT_THETA = jnp.array([2.0, 1.0, 1.0], dtype=jnp.float64)
WARMUP_MULTIPLIER = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare pathwise and REINFORCE gradient estimators for prior "
            "sensitivity in a Bayesian linear model."
        )
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run the full benchmark grid instead of the quick default.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Top-level seed used for deterministic dataset and chain generation.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1,
        help=(
            "Number of replicate keys to process together. The default is 1 to "
            "minimize memory use on large configurations."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON path where raw gradients and summaries are saved.",
    )
    parser.add_argument(
        "--dimensions",
        type=int,
        nargs="+",
        default=None,
        help="Optional dimensions D to run. Overrides the quick/full defaults.",
    )
    parser.add_argument(
        "--sample-multipliers",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Optional retained-sample multipliers m so each configuration uses "
            "M = m * D. Overrides the quick/full defaults."
        ),
    )
    parser.add_argument(
        "--num-repetitions",
        type=int,
        default=None,
        help="Optional number of repetitions. Overrides the quick/full defaults.",
    )
    return parser.parse_args()


def split_position(position: jax.Array, dim: int) -> tuple[jax.Array, jax.Array]:
    return position[:dim], position[dim]


def linear_model_log_joint(
    position: jax.Array, theta: jax.Array, dataset: Dataset
) -> jax.Array:
    x, y, _ = dataset
    dim = x.shape[1]
    w, log_alpha = split_position(position, dim)
    alpha = jnp.exp(log_alpha)
    a0, b0, lam = theta

    residual = y - x @ w
    n = x.shape[0]

    log_likelihood = (
        0.5 * n * (log_alpha - jnp.log(2.0 * jnp.pi))
        - 0.5 * alpha * jnp.sum(residual**2)
    )
    log_prior_w = 0.5 * dim * (jnp.log(lam) - jnp.log(2.0 * jnp.pi))
    log_prior_w = log_prior_w - 0.5 * lam * jnp.sum(w**2)
    log_prior_alpha = a0 * jnp.log(b0) - gammaln(a0) + a0 * log_alpha - b0 * alpha
    return log_likelihood + log_prior_w + log_prior_alpha


def functional_values(position: jax.Array, dim: int) -> jax.Array:
    w, log_alpha = split_position(position, dim)
    alpha = jnp.exp(log_alpha)
    return jnp.array([w[0], jnp.cos(alpha)])


def make_dataset(seed: int, dim: int) -> Dataset:
    key = jax.random.fold_in(jax.random.key(seed), dim)
    key_x, key_w, key_eps = jax.random.split(key, 3)

    n = 20 * dim
    x = jax.random.normal(key_x, (n, dim), dtype=jnp.float64) / jnp.sqrt(float(dim))
    w_true = jax.random.normal(key_w, (dim,), dtype=jnp.float64)
    alpha_true = jnp.array(2.0, dtype=jnp.float64)
    noise = jax.random.normal(key_eps, (n,), dtype=jnp.float64) / jnp.sqrt(alpha_true)
    y = x @ w_true + noise

    ridge = jnp.eye(dim, dtype=jnp.float64)
    xtx = x.T @ x
    xty = x.T @ y
    w_init = jnp.linalg.solve(xtx + ridge, xty)
    residual = y - x @ w_init
    residual_var = jnp.maximum(jnp.mean(residual**2), 1e-3)
    log_alpha_init = jnp.log(1.0 / residual_var)
    init_position = jnp.concatenate([w_init, jnp.array([log_alpha_init])])
    return Dataset(x=x, y=y, init_position=init_position)


def make_replicate_function(
    dataset: Dataset,
    warmup_steps: int,
    num_samples: int,
) -> Callable[[jax.Array, jax.Array], tuple[jax.Array, ReplicateAux]]:
    dim = dataset.x.shape[1]
    sampler = blackjax.reparameterized_slice(
        lambda position, theta: linear_model_log_joint(position, theta, dataset)
    )
    score_fn = jax.grad(
        lambda position, theta: linear_model_log_joint(position, theta, dataset),
        argnums=1,
    )

    def rollout_functionals(theta: jax.Array, replicate_key: jax.Array):
        warmup_key, sample_key = jax.random.split(replicate_key)
        warmup_keys = jax.random.split(warmup_key, warmup_steps)
        sample_keys = jax.random.split(sample_key, num_samples)

        state = sampler.init(dataset.init_position, theta)

        def warmup_body_fn(state, key):
            state, _ = sampler.step(key, state, theta)
            return state, None

        state, _ = jax.lax.scan(warmup_body_fn, state, warmup_keys)

        def sample_body_fn(carry, key):
            state, sum_functionals, sum_functional_score, sum_score = carry
            state, _ = sampler.step(key, state, theta)

            functionals = functional_values(state.position, dim)
            detached_position = jax.lax.stop_gradient(state.position)
            detached_functionals = jax.lax.stop_gradient(functionals)
            score = score_fn(detached_position, theta)

            sum_functionals = sum_functionals + functionals
            sum_functional_score = (
                sum_functional_score
                + detached_functionals[:, None] * score[None, :]
            )
            sum_score = sum_score + score
            return (state, sum_functionals, sum_functional_score, sum_score), None

        initial_carry = (
            state,
            jnp.zeros((2,), dtype=jnp.float64),
            jnp.zeros((2, 3), dtype=jnp.float64),
            jnp.zeros((3,), dtype=jnp.float64),
        )
        (state, sum_functionals, sum_functional_score, sum_score), _ = jax.lax.scan(
            sample_body_fn, initial_carry, sample_keys
        )
        del state

        mean_functionals = sum_functionals / num_samples
        aux = ReplicateAux(
            mean_functionals=jax.lax.stop_gradient(mean_functionals),
            mean_functional_score=sum_functional_score / num_samples,
            mean_score=sum_score / num_samples,
        )
        return mean_functionals, aux

    return jax.jit(jax.jacrev(rollout_functionals, argnums=0, has_aux=True))


def estimator_triplet(
    theta: jax.Array,
    replicate_key: jax.Array,
    replicate_function: Callable[[jax.Array, jax.Array], tuple[jax.Array, ReplicateAux]],
) -> jax.Array:
    pathwise_grad, aux = replicate_function(theta, replicate_key)
    raw = aux.mean_functional_score
    centered = raw - aux.mean_functionals[:, None] * aux.mean_score[None, :]
    stacked = jnp.stack((pathwise_grad, raw, centered), axis=0)
    return stacked


def run_configuration(
    seed: int,
    dim: int,
    num_samples: int,
    num_repetitions: int,
    chunk_size: int,
) -> np.ndarray:
    theta = jnp.array([2.0, 1.0, 1.0], dtype=jnp.float64)
    warmup_steps = WARMUP_MULTIPLIER * dim
    dataset = make_dataset(seed, dim)
    replicate_function = make_replicate_function(dataset, warmup_steps, num_samples)
    replicate_runner = jax.jit(
        lambda key: estimator_triplet(theta, key, replicate_function)
    )
    batch_runner = jax.jit(
        jax.vmap(lambda key: estimator_triplet(theta, key, replicate_function))
    )

    configuration_key = jax.random.fold_in(
        jax.random.key(seed), 10_000 * dim + num_samples
    )
    replicate_keys = jax.random.split(configuration_key, num_repetitions)

    outputs = []
    for start in range(0, num_repetitions, chunk_size):
        chunk_keys = replicate_keys[start : start + chunk_size]
        if chunk_keys.shape[0] == 1:
            outputs.append(np.asarray(replicate_runner(chunk_keys[0])))
        else:
            outputs.append(np.asarray(batch_runner(chunk_keys)))

    stacked = np.concatenate(
        [out[None, ...] if out.ndim == 3 else out for out in outputs],
        axis=0,
    )

    if not np.all(np.isfinite(stacked)):
        raise RuntimeError(
            f"Non-finite gradient estimates for D={dim}, M={num_samples}"
        )

    raw = stacked[:, 1]
    centered = stacked[:, 2]
    if np.allclose(raw, centered):
        raise RuntimeError(
            f"Centered REINFORCE unexpectedly matches raw REINFORCE for D={dim}, M={num_samples}"
        )

    return stacked


def summarize_results(gradient_estimates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(gradient_estimates, axis=0)
    variance = np.var(gradient_estimates, axis=0, ddof=1)
    return mean, variance


def make_mean_w1_runner(
    dataset: Dataset,
    warmup_steps: int,
    num_samples: int,
) -> Callable[[jax.Array, jax.Array], jax.Array]:
    sampler = blackjax.reparameterized_slice(
        lambda position, theta: linear_model_log_joint(position, theta, dataset)
    )

    def mean_w1(theta: jax.Array, rng_key: jax.Array) -> jax.Array:
        warmup_key, sample_key = jax.random.split(rng_key)
        warmup_keys = jax.random.split(warmup_key, warmup_steps)
        sample_keys = jax.random.split(sample_key, num_samples)

        state = sampler.init(dataset.init_position, theta)

        def warmup_body_fn(state, key):
            state, _ = sampler.step(key, state, theta)
            return state, None

        state, _ = jax.lax.scan(warmup_body_fn, state, warmup_keys)

        def sample_body_fn(carry, key):
            state, sum_w1 = carry
            state, _ = sampler.step(key, state, theta)
            return (state, sum_w1 + state.position[0]), None

        (_, sum_w1), _ = jax.lax.scan(
            sample_body_fn,
            (state, jnp.array(0.0, dtype=jnp.float64)),
            sample_keys,
        )
        return sum_w1 / num_samples

    return jax.jit(mean_w1)


def finite_difference_reference_gradient(
    seed: int,
    dim: int,
    *,
    num_samples: int = 20_000,
    relative_eps: float = 0.05,
) -> np.ndarray:
    dataset = make_dataset(seed, dim)
    warmup_steps = WARMUP_MULTIPLIER * dim
    mean_w1_runner = make_mean_w1_runner(dataset, warmup_steps, num_samples)
    base_key = jax.random.fold_in(jax.random.key(seed), 200_000 + dim)

    gradients = []
    for index in range(DEFAULT_THETA.shape[0]):
        eps = relative_eps * float(DEFAULT_THETA[index])
        shift = jnp.zeros_like(DEFAULT_THETA).at[index].set(eps)
        plus = mean_w1_runner(DEFAULT_THETA + shift, base_key)
        minus = mean_w1_runner(DEFAULT_THETA - shift, base_key)
        gradients.append((plus - minus) / (2.0 * eps))
    return np.asarray(jnp.stack(gradients))


def format_vector(x: np.ndarray) -> str:
    return np.array2string(
        np.asarray(x),
        precision=6,
        separator=", ",
        suppress_small=False,
        floatmode="maxprec_equal",
    )


def print_configuration_summary(
    dim: int,
    num_samples: int,
    num_repetitions: int,
    mean: np.ndarray,
    variance: np.ndarray,
) -> None:
    warmup_steps = WARMUP_MULTIPLIER * dim
    print(
        f"Configuration: D={dim}, num_samples={num_samples}, "
        f"warmup_steps={warmup_steps}, repetitions={num_repetitions}"
    )
    for functional_index, functional_name in enumerate(FUNCTIONAL_NAMES):
        print(f"  Functional: {functional_name}")
        for estimator_index, estimator_name in enumerate(ESTIMATOR_NAMES):
            print(f"    Estimator: {estimator_name}")
            print(f"      mean: {format_vector(mean[estimator_index, functional_index])}")
            print(
                f"      variance: {format_vector(variance[estimator_index, functional_index])}"
            )
    print()


def configuration_record(
    dim: int,
    num_samples: int,
    num_repetitions: int,
    gradient_estimates: np.ndarray,
    mean: np.ndarray,
    variance: np.ndarray,
) -> dict:
    return {
        "dimension": dim,
        "num_samples": num_samples,
        "warmup_steps": WARMUP_MULTIPLIER * dim,
        "num_repetitions": num_repetitions,
        "gradient_estimates": gradient_estimates.tolist(),
        "mean": mean.tolist(),
        "variance": variance.tolist(),
    }


def save_results(path: Path, args: argparse.Namespace, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": args.seed,
        "chunk_size": args.chunk_size,
        "full": bool(args.full),
        "dimensions": args.dimensions,
        "sample_multipliers": args.sample_multipliers,
        "num_repetitions": args.num_repetitions,
        "functional_names": list(FUNCTIONAL_NAMES),
        "estimator_names": list(ESTIMATOR_NAMES),
        "parameter_names": ["a0", "b0", "lambda"],
        "configurations": records,
    }
    path.write_text(json.dumps(payload, indent=2))


def benchmark_grid(args: argparse.Namespace) -> None:
    if args.chunk_size < 1:
        raise ValueError("--chunk-size must be at least 1")

    if args.dimensions is not None or args.sample_multipliers is not None:
        dimensions = args.dimensions if args.dimensions is not None else [5]
        multipliers = (
            args.sample_multipliers if args.sample_multipliers is not None else [100]
        )
        num_repetitions = (
            args.num_repetitions if args.num_repetitions is not None else 10
        )
        sample_counts = lambda dim: [multiplier * dim for multiplier in multipliers]
    elif args.full:
        dimensions = [5, 25, 125]
        num_repetitions = 200
        sample_counts = lambda dim: [100 * dim, 1000 * dim]
    else:
        dimensions = [5]
        num_repetitions = args.num_repetitions if args.num_repetitions is not None else 10
        sample_counts = lambda dim: [100 * dim]

    records = []
    for dim in dimensions:
        for num_samples in sample_counts(dim):
            print(
                f"Running D={dim}, M={num_samples}, repetitions={num_repetitions}, "
                f"chunk_size={args.chunk_size}",
                flush=True,
            )
            gradient_estimates = run_configuration(
                seed=args.seed,
                dim=dim,
                num_samples=num_samples,
                num_repetitions=num_repetitions,
                chunk_size=args.chunk_size,
            )
            mean, variance = summarize_results(gradient_estimates)
            print_configuration_summary(
                dim=dim,
                num_samples=num_samples,
                num_repetitions=num_repetitions,
                mean=mean,
                variance=variance,
            )
            records.append(
                configuration_record(
                    dim=dim,
                    num_samples=num_samples,
                    num_repetitions=num_repetitions,
                    gradient_estimates=gradient_estimates,
                    mean=mean,
                    variance=variance,
                )
            )

    if args.output is not None:
        save_results(args.output, args, records)
        print(f"Saved results to {args.output}")


def main() -> None:
    args = parse_args()
    benchmark_grid(args)


if __name__ == "__main__":
    main()
