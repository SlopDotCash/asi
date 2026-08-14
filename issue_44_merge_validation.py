"""Issue #44: Fix merge silently publishing comparison-less summary.

Ensures merge operation validates control arm presence before publishing.
"""

from typing import Dict, List, Any, Optional


class MergeValidator:
    """Validates merge operations have required controls."""

    @staticmethod
    def validate_control_presence(
        shards: Dict[str, Dict[str, Any]],
        control_name: str,
    ) -> tuple[bool, str]:
        """Validate named control exists in all shards.

        Returns (is_valid, error_message)
        """
        missing_shards = []

        for shard_name, shard_data in shards.items():
            arms = shard_data.get("arms", [])
            arm_names = [a.get("name") for a in arms]

            if control_name not in arm_names:
                missing_shards.append(shard_name)

        if missing_shards:
            return False, f"Control '{control_name}' missing in shards: {missing_shards}"

        return True, ""

    @staticmethod
    def generate_comparison_summary(
        shards: Dict[str, Dict[str, Any]],
        control_name: str,
        arms: List[str],
    ) -> Dict[str, Any]:
        """Generate summary ONLY after validating control presence.

        Refuses to generate if control is missing.
        """
        # VALIDATION: Check control exists
        is_valid, error_msg = MergeValidator.validate_control_presence(shards, control_name)

        if not is_valid:
            return {
                "valid": False,
                "error": error_msg,
                "summary": None,
                "comparison": None,
            }

        # Extract control performance
        control_perf = {}
        for shard_name, shard_data in shards.items():
            arm_data = next(
                (a for a in shard_data.get("arms", []) if a.get("name") == control_name),
                None
            )
            if arm_data:
                control_perf[shard_name] = arm_data.get("mean", 0)

        # Generate comparison
        comparison = {
            "control_name": control_name,
            "control_performance": control_perf,
            "arms": [],
        }

        for arm in arms:
            arm_perf = {}
            for shard_name, shard_data in shards.items():
                arm_data = next(
                    (a for a in shard_data.get("arms", []) if a.get("name") == arm),
                    None
                )
                if arm_data:
                    arm_perf[shard_name] = arm_data.get("mean", 0)

            if arm_perf:
                # Compute improvement over control
                avg_arm = __import__("numpy").mean(list(arm_perf.values()))
                avg_control = __import__("numpy").mean(list(control_perf.values()))
                improvement = (avg_arm - avg_control) / (avg_control + 1e-8)

                comparison["arms"].append({
                    "name": arm,
                    "performance": arm_perf,
                    "improvement_vs_control": float(improvement),
                })

        summary = {
            "valid": True,
            "error": None,
            "summary": {
                "control": control_name,
                "n_shards": len(shards),
                "n_arms_compared": len(arms),
            },
            "comparison": comparison,
        }

        return summary


class MergeOperationGuard:
    """Guards merge operations against invalid publishes."""

    @staticmethod
    def safe_publish_merge_results(
        shards: Dict[str, Dict[str, Any]],
        control_name: str,
        arms: List[str],
        publish_fn: Optional[callable] = None,
    ) -> Dict[str, Any]:
        """Safely publish merge results - refuses if control missing.

        Issue #44: Prevents silent publishing of comparison-less summaries.
        """
        # Validate
        summary = MergeValidator.generate_comparison_summary(shards, control_name, arms)

        if not summary["valid"]:
            return {
                "published": False,
                "error": summary["error"],
                "reason": "Control validation failed - refusing to publish",
            }

        # Publish only if valid
        if publish_fn:
            try:
                publish_fn(summary)
            except Exception as e:
                return {
                    "published": False,
                    "error": str(e),
                    "reason": "Publish function failed",
                }

        return {
            "published": True,
            "summary": summary["summary"],
            "comparison": summary["comparison"],
            "message": f"Merge published with valid control '{control_name}'",
        }


# Export for integration
MERGE_GUARDS = {
    "validate_control": MergeValidator.validate_control_presence,
    "safe_publish": MergeOperationGuard.safe_publish_merge_results,
}


def register_merge_guards():
    """Register merge operation guards (Issue #44)."""
    print("[OK] Registered merge guards (Issue #44)")
    return MERGE_GUARDS
