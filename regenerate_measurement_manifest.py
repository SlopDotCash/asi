"""Measurement manifest generator and validator.

Regenerates and validates the measurement manifest for all 150+ arms.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any
from unified_arm_registry import create_unified_registry


class MeasurementManifestGenerator:
    """Generate and manage measurement manifests."""

    def __init__(self, output_dir: str = "configs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.registry = create_unified_registry()

    def generate_manifest(self) -> Dict[str, Any]:
        """Generate complete measurement manifest."""
        return {
            "version": "2.0",
            "creation_date": "2026-08-15",
            "registry_version": self.registry.version,
            "total_arms": self.registry.get_summary()["total_arms"],
            "total_campaigns": self.registry.get_summary()["total_campaigns"],
            "total_estimated_hours": self.registry.get_summary()[
                "total_estimated_hours"
            ],
            "campaigns": self._generate_campaign_manifests(),
            "validation": self.registry.validate_all(),
        }

    def _generate_campaign_manifests(self) -> Dict[str, Any]:
        """Generate per-campaign manifests."""
        campaigns = {}

        for campaign_name in self.registry.campaign_index.keys():
            arms = self.registry.get_arms_by_campaign(campaign_name)
            total_hours = sum(arm.estimated_hours for arm in arms)

            campaigns[campaign_name] = {
                "name": campaign_name,
                "n_arms": len(arms),
                "total_hours": total_hours,
                "arms": [
                    {
                        "name": arm.name,
                        "description": arm.description,
                        "type": arm.arm_type,
                        "tags": arm.tags,
                        "estimated_hours": arm.estimated_hours,
                        "parameters": arm.parameters,
                    }
                    for arm in arms
                ],
            }

        return campaigns

    def export_manifest(self, manifest: Dict[str, Any]) -> Path:
        """Export manifest to JSON."""
        output_file = self.output_dir / "measurement_manifest.json"
        with open(output_file, "w") as f:
            json.dump(manifest, f, indent=2)
        return output_file

    def export_registry_summary(self) -> Path:
        """Export registry summary."""
        summary = self.registry.get_summary()
        output_file = self.output_dir / "registry_summary.json"
        with open(output_file, "w") as f:
            json.dump(summary, f, indent=2)
        return output_file

    def validate_manifest(self, manifest: Dict[str, Any]) -> Dict[str, Any]:
        """Validate manifest structure and content."""
        results = {
            "valid": True,
            "checks": {},
            "errors": [],
        }

        # Check required fields
        required_fields = [
            "version",
            "creation_date",
            "registry_version",
            "total_arms",
            "total_campaigns",
            "campaigns",
        ]
        for field in required_fields:
            if field not in manifest:
                results["valid"] = False
                results["errors"].append(f"Missing required field: {field}")
            results["checks"][f"has_{field}"] = field in manifest

        # Check campaign consistency
        if "campaigns" in manifest:
            expected_campaigns = set(self.registry.campaign_index.keys())
            actual_campaigns = set(manifest["campaigns"].keys())
            if expected_campaigns != actual_campaigns:
                results["valid"] = False
                results["errors"].append(
                    f"Campaign mismatch. Expected: {expected_campaigns}, "
                    f"Got: {actual_campaigns}"
                )
            results["checks"]["campaigns_match"] = (
                expected_campaigns == actual_campaigns
            )

        # Check arm counts
        total_arms = sum(c.get("n_arms", 0) for c in manifest["campaigns"].values())
        if total_arms != manifest.get("total_arms", 0):
            results["valid"] = False
            results["errors"].append(
                f"Arm count mismatch. Expected: {manifest['total_arms']}, "
                f"Got: {total_arms}"
            )
        results["checks"]["arm_count_consistent"] = (
            total_arms == manifest.get("total_arms", 0)
        )

        return results

    def generate_and_export_all(self) -> Dict[str, Path]:
        """Generate and export all manifests."""
        manifest = self.generate_manifest()
        manifest_file = self.export_manifest(manifest)
        summary_file = self.export_registry_summary()

        validation = self.validate_manifest(manifest)
        validation_file = self.output_dir / "manifest_validation.json"
        with open(validation_file, "w") as f:
            json.dump(validation, f, indent=2)

        return {
            "manifest": manifest_file,
            "summary": summary_file,
            "validation": validation_file,
        }


def regenerate_measurement_manifest(output_dir: str = "configs") -> None:
    """Regenerate measurement manifest."""
    generator = MeasurementManifestGenerator(output_dir)
    files = generator.generate_and_export_all()

    print("=== MEASUREMENT MANIFEST REGENERATED ===")
    for file_type, file_path in files.items():
        print(f"  {file_type}: {file_path}")

    # Validate
    with open(files["manifest"], "r") as f:
        manifest = json.load(f)

    validation = generator.validate_manifest(manifest)
    if validation["valid"]:
        print("\n[OK] Manifest validation PASSED")
    else:
        print("\n[ERROR] Manifest validation FAILED")
        for error in validation["errors"]:
            print(f"  - {error}")

    return files


if __name__ == "__main__":
    regenerate_measurement_manifest()
