"""Validation script for unified arm registry and backward compatibility.

Validates all arm registrations and ensures backward compatibility with
existing measurement configurations.
"""

import json
from typing import Dict, List, Any, Tuple
from pathlib import Path
from unified_arm_registry import create_unified_registry, ArmDefinition


class RegistryValidator:
    """Validate registry consistency and completeness."""

    def __init__(self):
        self.registry = create_unified_registry()
        self.validation_results = {
            "total_checks": 0,
            "passed": 0,
            "failed": 0,
            "warnings": 0,
            "details": {},
        }

    def validate_all(self) -> Dict[str, Any]:
        """Run all validation checks."""
        self._validate_registration_completeness()
        self._validate_arm_definitions()
        self._validate_campaign_consistency()
        self._validate_backward_compatibility()
        self._validate_resource_estimates()

        return self.validation_results

    def _validate_registration_completeness(self) -> None:
        """Check that all expected arms are registered."""
        check_name = "registration_completeness"
        self.validation_results["details"][check_name] = {
            "passed": True,
            "message": "All expected arms registered",
            "details": {},
        }

        expected_counts = {
            "ipmnist": 20,  # Consolidated variants
            "scr": 24,  # Consolidated variants
            "emnist": 29,  # Consolidated variants
            "micro_continual": 19,
            "forager": 19,
            "rule_discovery": 4,  # 4 phases representing 130 genomes
        }

        for campaign, expected_count in expected_counts.items():
            actual_count = len(
                self.registry.campaign_index.get(campaign, [])
            )
            passed = actual_count >= expected_count
            self.validation_results["details"][check_name]["details"][
                campaign
            ] = {
                "expected": expected_count,
                "actual": actual_count,
                "passed": passed,
            }

            if not passed:
                self.validation_results["details"][check_name]["passed"] = False
                self.validation_results["failed"] += 1
            else:
                self.validation_results["passed"] += 1

            self.validation_results["total_checks"] += 1

    def _validate_arm_definitions(self) -> None:
        """Validate individual arm definitions."""
        check_name = "arm_definitions"
        self.validation_results["details"][check_name] = {
            "passed": True,
            "message": "All arm definitions valid",
            "invalid_arms": [],
        }

        for arm_name, arm in self.registry.registry.items():
            try:
                self.registry._validate_arm(arm)
                self.validation_results["passed"] += 1
            except Exception as e:
                self.validation_results["details"][check_name][
                    "invalid_arms"
                ].append({"arm": arm_name, "error": str(e)})
                self.validation_results["details"][check_name]["passed"] = False
                self.validation_results["failed"] += 1

            self.validation_results["total_checks"] += 1

    def _validate_campaign_consistency(self) -> None:
        """Check consistency across campaigns."""
        check_name = "campaign_consistency"
        self.validation_results["details"][check_name] = {
            "passed": True,
            "message": "Campaign indices consistent",
            "inconsistencies": [],
        }

        # Verify bidirectional consistency
        for campaign, arm_names in self.registry.campaign_index.items():
            for arm_name in arm_names:
                if arm_name not in self.registry.registry:
                    self.validation_results["details"][check_name][
                        "inconsistencies"
                    ].append(
                        f"Campaign index references non-existent arm: {arm_name}"
                    )
                    self.validation_results["details"][check_name][
                        "passed"
                    ] = False
                    self.validation_results["failed"] += 1
                else:
                    arm = self.registry.registry[arm_name]
                    if arm.campaign != campaign:
                        self.validation_results["details"][check_name][
                            "inconsistencies"
                        ].append(
                            f"Arm {arm_name} campaign mismatch: "
                            f"index says {campaign}, arm says {arm.campaign}"
                        )
                        self.validation_results["details"][check_name][
                            "passed"
                        ] = False
                        self.validation_results["failed"] += 1
                    else:
                        self.validation_results["passed"] += 1

                self.validation_results["total_checks"] += 1

    def _validate_backward_compatibility(self) -> None:
        """Validate backward compatibility with legacy configs."""
        check_name = "backward_compatibility"
        self.validation_results["details"][check_name] = {
            "passed": True,
            "message": "Backward compatible with legacy configs",
            "legacy_arms": {},
        }

        # Legacy arm names from original configs
        legacy_arms = {
            "ipmnist": [
                "upgd_w_control",
                "adamw_control",
                "upgd_ema_norm",
            ],
            "scr": [
                "backprop_sgd_relu",
                "adamw_baseline",
                "upgd_w_baseline",
            ],
            "emnist": [
                "upgd_w",
                "adamw",
                "upgd_ema_norm_emnist",  # Renamed to avoid collision
            ],
            "micro_continual": [
                "rls_head_resid",
                "alignment_first",
                "naive_bayes_extended",
            ],
            "forager": [
                "dqn",
                "a3c",
                "horde",
            ],
        }

        for campaign, expected_arms in legacy_arms.items():
            campaign_arms = self.registry.campaign_index.get(campaign, [])
            self.validation_results["details"][check_name]["legacy_arms"][
                campaign
            ] = {
                "expected": expected_arms,
                "found": [a for a in expected_arms if a in campaign_arms],
                "missing": [a for a in expected_arms if a not in campaign_arms],
            }

            for arm in expected_arms:
                if arm in campaign_arms:
                    self.validation_results["passed"] += 1
                else:
                    self.validation_results["details"][check_name][
                        "passed"
                    ] = False
                    self.validation_results["failed"] += 1

                self.validation_results["total_checks"] += 1

    def _validate_resource_estimates(self) -> None:
        """Validate resource estimates."""
        check_name = "resource_estimates"
        self.validation_results["details"][check_name] = {
            "passed": True,
            "message": "Resource estimates valid",
            "campaign_hours": {},
            "total_hours": 0,
        }

        for campaign, arm_names in self.registry.campaign_index.items():
            total_hours = sum(
                self.registry.registry[arm].estimated_hours
                for arm in arm_names
            )
            self.validation_results["details"][check_name]["campaign_hours"][
                campaign
            ] = total_hours
            self.validation_results["details"][check_name][
                "total_hours"
            ] += total_hours

            # All campaigns should have positive estimates
            if total_hours > 0:
                self.validation_results["passed"] += 1
            else:
                self.validation_results["details"][check_name][
                    "passed"
                ] = False
                self.validation_results["failed"] += 1

            self.validation_results["total_checks"] += 1


def validate_registry() -> Tuple[bool, Dict[str, Any]]:
    """Validate the unified arm registry."""
    validator = RegistryValidator()
    results = validator.validate_all()

    return results["failed"] == 0, results


def print_validation_report(results: Dict[str, Any]) -> None:
    """Print validation report."""
    print("=" * 70)
    print("UNIFIED ARM REGISTRY VALIDATION REPORT")
    print("=" * 70)

    print(f"\nTotal checks: {results['total_checks']}")
    print(f"Passed: {results['passed']}")
    print(f"Failed: {results['failed']}")
    print(f"Warnings: {results['warnings']}")

    print("\n" + "-" * 70)
    print("DETAILED RESULTS")
    print("-" * 70)

    for check_name, details in results["details"].items():
        status = "[PASS]" if details.get("passed", False) else "[FAIL]"
        print(f"\n{status} {check_name}")
        print(f"  Message: {details.get('message', 'N/A')}")

        if "details" in details:
            for key, value in details["details"].items():
                if isinstance(value, dict):
                    print(f"  {key}: {json.dumps(value, indent=4)}")
                else:
                    print(f"  {key}: {value}")

        if "legacy_arms" in details:
            for campaign, info in details["legacy_arms"].items():
                print(f"  {campaign}:")
                print(f"    Expected: {info['expected']}")
                print(f"    Found: {info['found']}")
                if info["missing"]:
                    print(f"    Missing: {info['missing']}")

        if "campaign_hours" in details:
            for campaign, hours in details["campaign_hours"].items():
                print(f"  {campaign}: {hours:.1f} hours")
            print(f"  Total: {details['total_hours']:.1f} hours")

    print("\n" + "=" * 70)
    if results["failed"] == 0:
        print("[OK] VALIDATION PASSED - Registry is consistent and complete")
    else:
        print("[ERROR] VALIDATION FAILED - See errors above")
    print("=" * 70)


if __name__ == "__main__":
    passed, results = validate_registry()
    print_validation_report(results)

    # Export results
    with open("validation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    exit(0 if passed else 1)
