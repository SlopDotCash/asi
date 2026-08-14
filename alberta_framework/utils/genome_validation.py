"""Genome validation utilities for rule discovery genomes.

Validates rule genomes for correctness, interpretability, and fitness potential.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax.numpy as jnp
import numpy as np


class GenomeValidationResult(NamedTuple):
    """Result of genome validation."""

    is_valid: bool
    errors: list[str]
    warnings: list[str]
    metrics: dict[str, Any]


def validate_genome(
    genome: np.ndarray,
    genome_size: int = 256,
    flag_names: tuple[str, ...] = (),
) -> GenomeValidationResult:
    """Validate a rule genome for correctness and interpretability.

    Args:
        genome: Genome vector (float32)
        genome_size: Expected genome size
        flag_names: Valid flag names

    Returns:
        GenomeValidationResult with validation outcome
    """
    errors = []
    warnings = []
    metrics = {}

    # Check size
    if len(genome) != genome_size:
        errors.append(f"Genome size {len(genome)} != {genome_size}")

    # Check for NaN/Inf
    if np.any(~np.isfinite(genome)):
        errors.append("Genome contains NaN or Inf values")

    # Check value ranges
    if np.any(genome < -1.0) or np.any(genome > 1.0):
        warnings.append(f"Genome values outside [-1, 1]: min={np.min(genome):.4f}, max={np.max(genome):.4f}")

    # Check sparsity (how many flags are active)
    flag_count = np.sum(np.abs(genome[:len(flag_names)]) > 0.5)
    metrics["active_flags"] = int(flag_count)

    if flag_count > 10:
        warnings.append(f"Genome has {flag_count} active flags (>10) - may be over-specified")
    elif flag_count < 1:
        warnings.append("Genome has <1 active flags - may be under-specified")

    # Check parameter variance
    param_start = len(flag_names)
    param_variance = np.var(genome[param_start:])
    metrics["param_variance"] = float(param_variance)

    if param_variance < 0.01:
        warnings.append("Parameter variance is very low (<0.01) - limited diversity")

    # Estimate interpretability (lower is better - simpler rules)
    complexity = flag_count + np.sum(np.abs(genome[param_start:]) > 0.1)
    metrics["complexity_score"] = float(complexity)

    return GenomeValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        metrics=metrics,
    )


def validate_genome_population(
    genomes: np.ndarray,
    genome_size: int = 256,
) -> dict[str, Any]:
    """Validate a population of genomes.

    Args:
        genomes: Population matrix (n_genomes x genome_size)
        genome_size: Expected genome size

    Returns:
        Population validation report
    """
    n_genomes = genomes.shape[0]
    valid_count = 0
    error_genomes = []
    warning_genomes = []
    complexity_scores = []

    for i, genome in enumerate(genomes):
        result = validate_genome(genome, genome_size)
        if result.is_valid:
            valid_count += 1
        else:
            error_genomes.append(i)
        if result.warnings:
            warning_genomes.append(i)
        complexity_scores.append(result.metrics.get("complexity_score", 0))

    return {
        "n_genomes": n_genomes,
        "valid_genomes": valid_count,
        "validity_rate": float(valid_count / n_genomes),
        "error_genome_indices": error_genomes,
        "warning_genome_indices": warning_genomes,
        "avg_complexity": float(np.mean(complexity_scores)),
        "complexity_range": (float(np.min(complexity_scores)), float(np.max(complexity_scores))),
    }


def estimate_genome_fitness_potential(genome: np.ndarray, baseline_fitness: float = 0.88) -> float:
    """Estimate fitness potential of a genome based on its properties.

    Simple heuristic: genomes with moderate complexity and parameter variance
    are more likely to generalize well.

    Args:
        genome: Genome vector
        baseline_fitness: Expected baseline fitness (for scaling)

    Returns:
        Estimated fitness (0.7-1.0 range)
    """
    # Penalize extreme sparsity and density
    flag_count = np.sum(np.abs(genome[:50]) > 0.5)  # Assume first 50 are flags
    sparsity_penalty = abs(flag_count - 5) / 10  # Ideal: ~5 active flags

    # Reward parameter variance
    param_variance = np.var(genome[50:])
    variance_bonus = min(param_variance / 0.1, 0.2)  # Max +0.2

    # Reward avoiding extreme values
    extreme_penalty = np.mean(np.abs(np.clip(genome, -1, 1) - genome))

    estimated_fitness = baseline_fitness - sparsity_penalty + variance_bonus - extreme_penalty * 0.1
    return float(np.clip(estimated_fitness, 0.7, 1.0))


def genome_diversity_score(genomes: np.ndarray) -> float:
    """Compute population diversity via pairwise L2 distance.

    Args:
        genomes: Population matrix

    Returns:
        Average pairwise normalized L2 distance (0-1, higher = more diverse)
    """
    if len(genomes) < 2:
        return 0.0

    distances = []
    for i in range(min(len(genomes), 10)):  # Sample first 10 to avoid O(n^2)
        for j in range(i + 1, min(len(genomes), 10)):
            dist = np.linalg.norm(genomes[i] - genomes[j])
            normalized_dist = dist / (np.sqrt(2) * np.sqrt(len(genomes[i])))
            distances.append(normalized_dist)

    return float(np.mean(distances)) if distances else 0.0
