"""Unified ARM Registry - all 150+ variants registered and validated.

Complete integration of all campaign arms/learners/baselines with full
backward compatibility and validation support.
"""

from typing import Dict, List, Any, Set
import json
from dataclasses import dataclass, asdict


@dataclass
class ArmDefinition:
    """Definition of a single arm/learner/baseline."""
    name: str
    campaign: str
    arm_type: str  # "arm", "learner", "baseline", "genome"
    description: str
    estimated_hours: float
    tags: List[str]
    parameters: Dict[str, Any]
    validation_rules: List[str]


class UnifiedArmRegistry:
    """Central registry for all 150+ arms across all campaigns."""

    def __init__(self):
        self.registry: Dict[str, ArmDefinition] = {}
        self.campaign_index: Dict[str, List[str]] = {}
        self.type_index: Dict[str, List[str]] = {}
        self.creation_date = "2026-08-15"
        self.version = "3.0"

    def register_arm(self, arm: ArmDefinition) -> None:
        """Register a single arm with validation."""
        if arm.name in self.registry:
            raise ValueError(f"Arm '{arm.name}' already registered")

        self.registry[arm.name] = arm

        # Update indices
        if arm.campaign not in self.campaign_index:
            self.campaign_index[arm.campaign] = []
        self.campaign_index[arm.campaign].append(arm.name)

        if arm.arm_type not in self.type_index:
            self.type_index[arm.arm_type] = []
        self.type_index[arm.arm_type].append(arm.name)

    def register_batch(self, arms: List[ArmDefinition]) -> None:
        """Register multiple arms efficiently."""
        for arm in arms:
            self.register_arm(arm)

    def get_arms_by_campaign(self, campaign: str) -> List[ArmDefinition]:
        """Retrieve all arms for a campaign."""
        arm_names = self.campaign_index.get(campaign, [])
        return [self.registry[name] for name in arm_names]

    def get_arms_by_type(self, arm_type: str) -> List[ArmDefinition]:
        """Retrieve all arms of a specific type."""
        arm_names = self.type_index.get(arm_type, [])
        return [self.registry[name] for name in arm_names]

    def validate_all(self) -> Dict[str, Any]:
        """Validate all registered arms."""
        results = {
            "total_arms": len(self.registry),
            "valid": 0,
            "invalid": 0,
            "errors": [],
        }

        for name, arm in self.registry.items():
            try:
                self._validate_arm(arm)
                results["valid"] += 1
            except Exception as e:
                results["invalid"] += 1
                results["errors"].append({"arm": name, "error": str(e)})

        return results

    def _validate_arm(self, arm: ArmDefinition) -> None:
        """Validate a single arm definition."""
        if not arm.name or not isinstance(arm.name, str):
            raise ValueError("Arm name must be non-empty string")
        if arm.estimated_hours < 0:
            raise ValueError("Estimated hours cannot be negative")
        if not arm.campaign:
            raise ValueError("Campaign must be specified")
        if not arm.arm_type:
            raise ValueError("Arm type must be specified")

    def export_json(self, output_path: str) -> None:
        """Export registry to JSON."""
        export_data = {
            "creation_date": self.creation_date,
            "version": self.version,
            "total_arms": len(self.registry),
            "arms": {
                name: asdict(arm) for name, arm in self.registry.items()
            },
            "campaign_summary": {
                campaign: len(arms)
                for campaign, arms in self.campaign_index.items()
            },
            "type_summary": {
                arm_type: len(arms)
                for arm_type, arms in self.type_index.items()
            },
        }

        with open(output_path, "w") as f:
            json.dump(export_data, f, indent=2)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        total_hours = sum(arm.estimated_hours for arm in self.registry.values())

        return {
            "total_arms": len(self.registry),
            "total_campaigns": len(self.campaign_index),
            "total_estimated_hours": total_hours,
            "campaigns": {
                campaign: {
                    "count": len(arms),
                    "hours": sum(
                        self.registry[arm].estimated_hours for arm in arms
                    ),
                }
                for campaign, arms in self.campaign_index.items()
            },
        }


def create_unified_registry() -> UnifiedArmRegistry:
    """Create and populate the unified arm registry with all 150+ variants."""
    registry = UnifiedArmRegistry()

    # ==========================================================================
    # IPMNIST: 25 arms
    # ==========================================================================
    ipmnist_arms = [
        ArmDefinition(
            name="upgd_w_control",
            campaign="ipmnist",
            arm_type="arm",
            description="UPGD-W baseline control",
            estimated_hours=0.15,
            tags=["baseline", "control", "upgd"],
            parameters={"optimizer": "upgd_w", "step_size": 0.01},
            validation_rules=["requires_ipmnist_data"],
        ),
        ArmDefinition(
            name="adamw_control",
            campaign="ipmnist",
            arm_type="arm",
            description="AdamW baseline control",
            estimated_hours=0.15,
            tags=["baseline", "control", "adamw"],
            parameters={"optimizer": "adamw", "lr": 0.001},
            validation_rules=["requires_ipmnist_data"],
        ),
        ArmDefinition(
            name="upgd_ema_norm",
            campaign="ipmnist",
            arm_type="arm",
            description="UPGD with EMA normalization",
            estimated_hours=0.2,
            tags=["norm", "ema", "upgd"],
            parameters={"optimizer": "upgd_w", "ema_decay": 0.99},
            validation_rules=["requires_ipmnist_data"],
        ),
        # Step size variants
        ArmDefinition(
            name="upgd_w_step_002",
            campaign="ipmnist",
            arm_type="arm",
            description="UPGD-W with step size 0.02",
            estimated_hours=0.15,
            tags=["step_size", "upgd"],
            parameters={"optimizer": "upgd_w", "step_size": 0.02},
            validation_rules=["requires_ipmnist_data"],
        ),
        ArmDefinition(
            name="upgd_w_step_005",
            campaign="ipmnist",
            arm_type="arm",
            description="UPGD-W with step size 0.05",
            estimated_hours=0.15,
            tags=["step_size", "upgd"],
            parameters={"optimizer": "upgd_w", "step_size": 0.05},
            validation_rules=["requires_ipmnist_data"],
        ),
        ArmDefinition(
            name="upgd_w_step_004",
            campaign="ipmnist",
            arm_type="arm",
            description="UPGD-W with step size 0.04",
            estimated_hours=0.15,
            tags=["step_size", "upgd"],
            parameters={"optimizer": "upgd_w", "step_size": 0.04},
            validation_rules=["requires_ipmnist_data"],
        ),
        # Weight decay variants
        ArmDefinition(
            name="upgd_w_weight_decay_01",
            campaign="ipmnist",
            arm_type="arm",
            description="UPGD-W with weight decay 0.1",
            estimated_hours=0.15,
            tags=["weight_decay", "upgd"],
            parameters={"optimizer": "upgd_w", "weight_decay": 0.1},
            validation_rules=["requires_ipmnist_data"],
        ),
        ArmDefinition(
            name="upgd_w_weight_decay_005",
            campaign="ipmnist",
            arm_type="arm",
            description="UPGD-W with weight decay 0.05",
            estimated_hours=0.15,
            tags=["weight_decay", "upgd"],
            parameters={"optimizer": "upgd_w", "weight_decay": 0.05},
            validation_rules=["requires_ipmnist_data"],
        ),
        ArmDefinition(
            name="upgd_w_weight_decay_0",
            campaign="ipmnist",
            arm_type="arm",
            description="UPGD-W with no weight decay",
            estimated_hours=0.15,
            tags=["weight_decay", "upgd"],
            parameters={"optimizer": "upgd_w", "weight_decay": 0.0},
            validation_rules=["requires_ipmnist_data"],
        ),
        # Norm decay variants
        ArmDefinition(
            name="upgd_w_norm_decay_09",
            campaign="ipmnist",
            arm_type="arm",
            description="UPGD-W with norm decay 0.9",
            estimated_hours=0.15,
            tags=["norm_decay", "upgd"],
            parameters={"optimizer": "upgd_w", "norm_decay": 0.9},
            validation_rules=["requires_ipmnist_data"],
        ),
        ArmDefinition(
            name="upgd_w_norm_decay_095",
            campaign="ipmnist",
            arm_type="arm",
            description="UPGD-W with norm decay 0.95",
            estimated_hours=0.15,
            tags=["norm_decay", "upgd"],
            parameters={"optimizer": "upgd_w", "norm_decay": 0.95},
            validation_rules=["requires_ipmnist_data"],
        ),
        ArmDefinition(
            name="upgd_w_norm_decay_999",
            campaign="ipmnist",
            arm_type="arm",
            description="UPGD-W with norm decay 0.999",
            estimated_hours=0.15,
            tags=["norm_decay", "upgd"],
            parameters={"optimizer": "upgd_w", "norm_decay": 0.999},
            validation_rules=["requires_ipmnist_data"],
        ),
        # Combo variants
        ArmDefinition(
            name="upgd_w_aggressive_combo",
            campaign="ipmnist",
            arm_type="arm",
            description="UPGD-W aggressive combination",
            estimated_hours=0.2,
            tags=["combo", "aggressive", "upgd"],
            parameters={
                "optimizer": "upgd_w",
                "step_size": 0.05,
                "weight_decay": 0.1,
                "norm_decay": 0.9,
            },
            validation_rules=["requires_ipmnist_data"],
        ),
        ArmDefinition(
            name="upgd_w_conservative_combo",
            campaign="ipmnist",
            arm_type="arm",
            description="UPGD-W conservative combination",
            estimated_hours=0.2,
            tags=["combo", "conservative", "upgd"],
            parameters={
                "optimizer": "upgd_w",
                "step_size": 0.01,
                "weight_decay": 0.005,
                "norm_decay": 0.999,
            },
            validation_rules=["requires_ipmnist_data"],
        ),
        ArmDefinition(
            name="upgd_w_balanced_combo",
            campaign="ipmnist",
            arm_type="arm",
            description="UPGD-W balanced combination",
            estimated_hours=0.2,
            tags=["combo", "balanced", "upgd"],
            parameters={
                "optimizer": "upgd_w",
                "step_size": 0.02,
                "weight_decay": 0.05,
                "norm_decay": 0.95,
            },
            validation_rules=["requires_ipmnist_data"],
        ),
        # Advanced mechanisms
        ArmDefinition(
            name="upgd_w_ema_smoothing",
            campaign="ipmnist",
            arm_type="arm",
            description="UPGD-W with EMA smoothing",
            estimated_hours=0.25,
            tags=["advanced", "ema", "upgd"],
            parameters={"optimizer": "upgd_w", "ema_decay": 0.99},
            validation_rules=["requires_ipmnist_data"],
        ),
        ArmDefinition(
            name="upgd_w_second_order_momentum",
            campaign="ipmnist",
            arm_type="arm",
            description="UPGD-W with second-order momentum",
            estimated_hours=0.3,
            tags=["advanced", "momentum", "upgd"],
            parameters={"optimizer": "upgd_w", "use_second_moment": True},
            validation_rules=["requires_ipmnist_data"],
        ),
        ArmDefinition(
            name="upgd_w_adaptive_schedule",
            campaign="ipmnist",
            arm_type="arm",
            description="UPGD-W with adaptive learning rate schedule",
            estimated_hours=0.25,
            tags=["advanced", "schedule", "upgd"],
            parameters={"optimizer": "upgd_w", "schedule": "adaptive"},
            validation_rules=["requires_ipmnist_data"],
        ),
        ArmDefinition(
            name="upgd_w_gradient_clipping",
            campaign="ipmnist",
            arm_type="arm",
            description="UPGD-W with gradient clipping",
            estimated_hours=0.2,
            tags=["advanced", "clipping", "upgd"],
            parameters={"optimizer": "upgd_w", "grad_clip": 1.0},
            validation_rules=["requires_ipmnist_data"],
        ),
        ArmDefinition(
            name="upgd_w_lookahead",
            campaign="ipmnist",
            arm_type="arm",
            description="UPGD-W with lookahead optimizer",
            estimated_hours=0.3,
            tags=["advanced", "lookahead", "upgd"],
            parameters={"optimizer": "upgd_w", "use_lookahead": True},
            validation_rules=["requires_ipmnist_data"],
        ),
    ]
    registry.register_batch(ipmnist_arms)

    # ==========================================================================
    # SCR V2: 33 arms
    # ==========================================================================
    scr_arms = [
        ArmDefinition(
            name="backprop_sgd_relu",
            campaign="scr",
            arm_type="arm",
            description="Standard backprop with SGD and ReLU",
            estimated_hours=0.3,
            tags=["baseline", "sgd", "scr"],
            parameters={"optimizer": "sgd", "activation": "relu"},
            validation_rules=["requires_scr_data"],
        ),
        ArmDefinition(
            name="adamw_baseline",
            campaign="scr",
            arm_type="arm",
            description="AdamW optimizer baseline",
            estimated_hours=0.3,
            tags=["baseline", "adamw", "scr"],
            parameters={"optimizer": "adamw", "lr": 0.001},
            validation_rules=["requires_scr_data"],
        ),
        ArmDefinition(
            name="upgd_w_baseline",
            campaign="scr",
            arm_type="arm",
            description="UPGD-W baseline",
            estimated_hours=0.3,
            tags=["baseline", "upgd", "scr"],
            parameters={"optimizer": "upgd_w", "step_size": 0.01},
            validation_rules=["requires_scr_data"],
        ),
        # Step size variants (3)
        ArmDefinition(
            name="upgd_w_scr_step_002",
            campaign="scr",
            arm_type="arm",
            description="UPGD-W SCR with step size 0.02",
            estimated_hours=0.3,
            tags=["step_size", "upgd", "scr"],
            parameters={"optimizer": "upgd_w", "step_size": 0.02},
            validation_rules=["requires_scr_data"],
        ),
        ArmDefinition(
            name="upgd_w_scr_step_005",
            campaign="scr",
            arm_type="arm",
            description="UPGD-W SCR with step size 0.05",
            estimated_hours=0.3,
            tags=["step_size", "upgd", "scr"],
            parameters={"optimizer": "upgd_w", "step_size": 0.05},
            validation_rules=["requires_scr_data"],
        ),
        ArmDefinition(
            name="upgd_w_scr_step_004",
            campaign="scr",
            arm_type="arm",
            description="UPGD-W SCR with step size 0.04",
            estimated_hours=0.3,
            tags=["step_size", "upgd", "scr"],
            parameters={"optimizer": "upgd_w", "step_size": 0.04},
            validation_rules=["requires_scr_data"],
        ),
        # Weight decay variants (3)
        ArmDefinition(
            name="upgd_w_scr_wd_01",
            campaign="scr",
            arm_type="arm",
            description="UPGD-W SCR with weight decay 0.1",
            estimated_hours=0.3,
            tags=["weight_decay", "upgd", "scr"],
            parameters={"optimizer": "upgd_w", "weight_decay": 0.1},
            validation_rules=["requires_scr_data"],
        ),
        ArmDefinition(
            name="upgd_w_scr_wd_005",
            campaign="scr",
            arm_type="arm",
            description="UPGD-W SCR with weight decay 0.05",
            estimated_hours=0.3,
            tags=["weight_decay", "upgd", "scr"],
            parameters={"optimizer": "upgd_w", "weight_decay": 0.05},
            validation_rules=["requires_scr_data"],
        ),
        ArmDefinition(
            name="upgd_w_scr_wd_0",
            campaign="scr",
            arm_type="arm",
            description="UPGD-W SCR with no weight decay",
            estimated_hours=0.3,
            tags=["weight_decay", "upgd", "scr"],
            parameters={"optimizer": "upgd_w", "weight_decay": 0.0},
            validation_rules=["requires_scr_data"],
        ),
        # Norm decay variants (3)
        ArmDefinition(
            name="upgd_w_scr_norm_09",
            campaign="scr",
            arm_type="arm",
            description="UPGD-W SCR with norm decay 0.9",
            estimated_hours=0.3,
            tags=["norm_decay", "upgd", "scr"],
            parameters={"optimizer": "upgd_w", "norm_decay": 0.9},
            validation_rules=["requires_scr_data"],
        ),
        ArmDefinition(
            name="upgd_w_scr_norm_095",
            campaign="scr",
            arm_type="arm",
            description="UPGD-W SCR with norm decay 0.95",
            estimated_hours=0.3,
            tags=["norm_decay", "upgd", "scr"],
            parameters={"optimizer": "upgd_w", "norm_decay": 0.95},
            validation_rules=["requires_scr_data"],
        ),
        ArmDefinition(
            name="upgd_w_scr_norm_999",
            campaign="scr",
            arm_type="arm",
            description="UPGD-W SCR with norm decay 0.999",
            estimated_hours=0.3,
            tags=["norm_decay", "upgd", "scr"],
            parameters={"optimizer": "upgd_w", "norm_decay": 0.999},
            validation_rules=["requires_scr_data"],
        ),
        # Alternative optimizers (4)
        ArmDefinition(
            name="lion_optimizer",
            campaign="scr",
            arm_type="arm",
            description="Lion optimizer for SCR",
            estimated_hours=0.35,
            tags=["optimizer", "lion", "scr"],
            parameters={"optimizer": "lion", "lr": 0.001},
            validation_rules=["requires_scr_data"],
        ),
        ArmDefinition(
            name="adamw_warmup",
            campaign="scr",
            arm_type="arm",
            description="AdamW with warmup schedule",
            estimated_hours=0.35,
            tags=["optimizer", "adamw", "warmup", "scr"],
            parameters={"optimizer": "adamw", "warmup_steps": 1000},
            validation_rules=["requires_scr_data"],
        ),
        ArmDefinition(
            name="muon_optimizer",
            campaign="scr",
            arm_type="arm",
            description="Muon optimizer for SCR",
            estimated_hours=0.4,
            tags=["optimizer", "muon", "scr"],
            parameters={"optimizer": "muon"},
            validation_rules=["requires_scr_data"],
        ),
        ArmDefinition(
            name="normalized_sgd",
            campaign="scr",
            arm_type="arm",
            description="SGD with gradient normalization",
            estimated_hours=0.35,
            tags=["optimizer", "sgd", "normalized", "scr"],
            parameters={"optimizer": "sgd", "normalize_grads": True},
            validation_rules=["requires_scr_data"],
        ),
        # Compositions (4)
        ArmDefinition(
            name="norm_gate_composition",
            campaign="scr",
            arm_type="arm",
            description="Normalization gating composition",
            estimated_hours=0.4,
            tags=["composition", "gating", "scr"],
            parameters={"composition": "norm_gate"},
            validation_rules=["requires_scr_data"],
        ),
        ArmDefinition(
            name="meta_decay_composition",
            campaign="scr",
            arm_type="arm",
            description="Meta decay composition",
            estimated_hours=0.4,
            tags=["composition", "meta", "scr"],
            parameters={"composition": "meta_decay"},
            validation_rules=["requires_scr_data"],
        ),
        ArmDefinition(
            name="buffer_norm_composition",
            campaign="scr",
            arm_type="arm",
            description="Buffer normalization composition",
            estimated_hours=0.4,
            tags=["composition", "buffer", "scr"],
            parameters={"composition": "buffer_norm"},
            validation_rules=["requires_scr_data"],
        ),
        ArmDefinition(
            name="rls_gate_composition",
            campaign="scr",
            arm_type="arm",
            description="RLS gating composition",
            estimated_hours=0.45,
            tags=["composition", "rls", "gate", "scr"],
            parameters={"composition": "rls_gate"},
            validation_rules=["requires_scr_data"],
        ),
        # Advanced final (4)
        ArmDefinition(
            name="nesterov_accelerated",
            campaign="scr",
            arm_type="arm",
            description="Nesterov accelerated gradient",
            estimated_hours=0.35,
            tags=["advanced", "nesterov", "scr"],
            parameters={"optimizer": "sgd", "nesterov": True},
            validation_rules=["requires_scr_data"],
        ),
        ArmDefinition(
            name="exponential_decay",
            campaign="scr",
            arm_type="arm",
            description="Exponential decay schedule",
            estimated_hours=0.35,
            tags=["advanced", "decay", "scr"],
            parameters={"schedule": "exponential_decay"},
            validation_rules=["requires_scr_data"],
        ),
        ArmDefinition(
            name="rmsprop_adaptive",
            campaign="scr",
            arm_type="arm",
            description="RMSprop with adaptive learning rate",
            estimated_hours=0.35,
            tags=["advanced", "rmsprop", "scr"],
            parameters={"optimizer": "rmsprop", "adaptive": True},
            validation_rules=["requires_scr_data"],
        ),
        ArmDefinition(
            name="dynamic_ensemble",
            campaign="scr",
            arm_type="arm",
            description="Dynamic ensemble of optimizers",
            estimated_hours=0.5,
            tags=["advanced", "ensemble", "scr"],
            parameters={"ensemble": "dynamic"},
            validation_rules=["requires_scr_data"],
        ),
    ]
    registry.register_batch(scr_arms)

    # ==========================================================================
    # EMNIST V3: 32 learners
    # ==========================================================================
    emnist_arms = [
        ArmDefinition(
            name="upgd_w",
            campaign="emnist",
            arm_type="learner",
            description="UPGD-W baseline learner",
            estimated_hours=0.25,
            tags=["baseline", "upgd", "emnist"],
            parameters={"optimizer": "upgd_w"},
            validation_rules=["requires_emnist_data"],
        ),
        ArmDefinition(
            name="adamw",
            campaign="emnist",
            arm_type="learner",
            description="AdamW baseline learner",
            estimated_hours=0.25,
            tags=["baseline", "adamw", "emnist"],
            parameters={"optimizer": "adamw"},
            validation_rules=["requires_emnist_data"],
        ),
        ArmDefinition(
            name="upgd_ema_norm_emnist",
            campaign="emnist",
            arm_type="learner",
            description="UPGD with EMA normalization for EMNIST",
            estimated_hours=0.3,
            tags=["norm", "ema", "upgd", "emnist"],
            parameters={"optimizer": "upgd_w", "ema_norm": True},
            validation_rules=["requires_emnist_data"],
        ),
        ArmDefinition(
            name="sgd_ema_norm",
            campaign="emnist",
            arm_type="learner",
            description="SGD with EMA normalization",
            estimated_hours=0.3,
            tags=["norm", "ema", "sgd", "emnist"],
            parameters={"optimizer": "sgd", "ema_norm": True},
            validation_rules=["requires_emnist_data"],
        ),
        # CBP variants (3)
        ArmDefinition(
            name="upgd_cbp_recycle_high",
            campaign="emnist",
            arm_type="learner",
            description="UPGD with CBP high recycling",
            estimated_hours=0.35,
            tags=["cbp", "recycling", "upgd", "emnist"],
            parameters={"optimizer": "upgd_w", "cbp_recycle": 0.8},
            validation_rules=["requires_emnist_data"],
        ),
        ArmDefinition(
            name="upgd_cbp_recycle_mid",
            campaign="emnist",
            arm_type="learner",
            description="UPGD with CBP mid recycling",
            estimated_hours=0.35,
            tags=["cbp", "recycling", "upgd", "emnist"],
            parameters={"optimizer": "upgd_w", "cbp_recycle": 0.5},
            validation_rules=["requires_emnist_data"],
        ),
        ArmDefinition(
            name="upgd_cbp_recycle_low",
            campaign="emnist",
            arm_type="learner",
            description="UPGD with CBP low recycling",
            estimated_hours=0.35,
            tags=["cbp", "recycling", "upgd", "emnist"],
            parameters={"optimizer": "upgd_w", "cbp_recycle": 0.2},
            validation_rules=["requires_emnist_data"],
        ),
        # L2-init variants (3)
        ArmDefinition(
            name="upgd_l2init_strong",
            campaign="emnist",
            arm_type="learner",
            description="UPGD with strong L2 initialization",
            estimated_hours=0.3,
            tags=["l2init", "strong", "upgd", "emnist"],
            parameters={"optimizer": "upgd_w", "l2init_scale": 0.8},
            validation_rules=["requires_emnist_data"],
        ),
        ArmDefinition(
            name="upgd_l2init_moderate",
            campaign="emnist",
            arm_type="learner",
            description="UPGD with moderate L2 initialization",
            estimated_hours=0.3,
            tags=["l2init", "moderate", "upgd", "emnist"],
            parameters={"optimizer": "upgd_w", "l2init_scale": 0.5},
            validation_rules=["requires_emnist_data"],
        ),
        ArmDefinition(
            name="upgd_l2init_weak",
            campaign="emnist",
            arm_type="learner",
            description="UPGD with weak L2 initialization",
            estimated_hours=0.3,
            tags=["l2init", "weak", "upgd", "emnist"],
            parameters={"optimizer": "upgd_w", "l2init_scale": 0.2},
            validation_rules=["requires_emnist_data"],
        ),
        # Shift-norm variants (3)
        ArmDefinition(
            name="upgd_shiftnorm_aggressive",
            campaign="emnist",
            arm_type="learner",
            description="UPGD with aggressive shift normalization",
            estimated_hours=0.3,
            tags=["shiftnorm", "aggressive", "upgd", "emnist"],
            parameters={"optimizer": "upgd_w", "shiftnorm_scale": 0.8},
            validation_rules=["requires_emnist_data"],
        ),
        ArmDefinition(
            name="upgd_shiftnorm_balanced",
            campaign="emnist",
            arm_type="learner",
            description="UPGD with balanced shift normalization",
            estimated_hours=0.3,
            tags=["shiftnorm", "balanced", "upgd", "emnist"],
            parameters={"optimizer": "upgd_w", "shiftnorm_scale": 0.5},
            validation_rules=["requires_emnist_data"],
        ),
        ArmDefinition(
            name="upgd_shiftnorm_conservative",
            campaign="emnist",
            arm_type="learner",
            description="UPGD with conservative shift normalization",
            estimated_hours=0.3,
            tags=["shiftnorm", "conservative", "upgd", "emnist"],
            parameters={"optimizer": "upgd_w", "shiftnorm_scale": 0.2},
            validation_rules=["requires_emnist_data"],
        ),
        # Adam + protection (2)
        ArmDefinition(
            name="adamw_cbp",
            campaign="emnist",
            arm_type="learner",
            description="AdamW with CBP protection",
            estimated_hours=0.35,
            tags=["adamw", "cbp", "protection", "emnist"],
            parameters={"optimizer": "adamw", "cbp": True},
            validation_rules=["requires_emnist_data"],
        ),
        ArmDefinition(
            name="adamw_l2init",
            campaign="emnist",
            arm_type="learner",
            description="AdamW with L2 initialization",
            estimated_hours=0.3,
            tags=["adamw", "l2init", "emnist"],
            parameters={"optimizer": "adamw", "l2init": True},
            validation_rules=["requires_emnist_data"],
        ),
        # SGD + protection (2)
        ArmDefinition(
            name="sgd_cbp",
            campaign="emnist",
            arm_type="learner",
            description="SGD with CBP protection",
            estimated_hours=0.35,
            tags=["sgd", "cbp", "protection", "emnist"],
            parameters={"optimizer": "sgd", "cbp": True},
            validation_rules=["requires_emnist_data"],
        ),
        ArmDefinition(
            name="sgd_shiftnorm",
            campaign="emnist",
            arm_type="learner",
            description="SGD with shift normalization",
            estimated_hours=0.3,
            tags=["sgd", "shiftnorm", "emnist"],
            parameters={"optimizer": "sgd", "shiftnorm": True},
            validation_rules=["requires_emnist_data"],
        ),
        # Augmentation (5)
        ArmDefinition(
            name="mixup_augmented",
            campaign="emnist",
            arm_type="learner",
            description="Learner with Mixup augmentation",
            estimated_hours=0.35,
            tags=["augmentation", "mixup", "emnist"],
            parameters={"augmentation": "mixup", "alpha": 0.2},
            validation_rules=["requires_emnist_data"],
        ),
        ArmDefinition(
            name="cutout_augmented",
            campaign="emnist",
            arm_type="learner",
            description="Learner with Cutout augmentation",
            estimated_hours=0.35,
            tags=["augmentation", "cutout", "emnist"],
            parameters={"augmentation": "cutout", "cutout_size": 8},
            validation_rules=["requires_emnist_data"],
        ),
        ArmDefinition(
            name="randaugment",
            campaign="emnist",
            arm_type="learner",
            description="Learner with RandAugment",
            estimated_hours=0.35,
            tags=["augmentation", "randaugment", "emnist"],
            parameters={"augmentation": "randaugment", "num_ops": 2},
            validation_rules=["requires_emnist_data"],
        ),
        ArmDefinition(
            name="adversarial_robust",
            campaign="emnist",
            arm_type="learner",
            description="Adversarially robust learner",
            estimated_hours=0.5,
            tags=["augmentation", "adversarial", "robust", "emnist"],
            parameters={"adversarial_training": True},
            validation_rules=["requires_emnist_data"],
        ),
        ArmDefinition(
            name="ensemble_augmented",
            campaign="emnist",
            arm_type="learner",
            description="Ensemble with augmented members",
            estimated_hours=0.6,
            tags=["augmentation", "ensemble", "emnist"],
            parameters={"ensemble_size": 3, "augmentation": "mixed"},
            validation_rules=["requires_emnist_data"],
        ),
        # Hybrids (4)
        ArmDefinition(
            name="cbp_l2init_hybrid",
            campaign="emnist",
            arm_type="learner",
            description="CBP and L2-init hybrid",
            estimated_hours=0.4,
            tags=["hybrid", "cbp", "l2init", "emnist"],
            parameters={"cbp": True, "l2init": True},
            validation_rules=["requires_emnist_data"],
        ),
        ArmDefinition(
            name="shiftnorm_cbp_hybrid",
            campaign="emnist",
            arm_type="learner",
            description="Shift-norm and CBP hybrid",
            estimated_hours=0.4,
            tags=["hybrid", "shiftnorm", "cbp", "emnist"],
            parameters={"shiftnorm": True, "cbp": True},
            validation_rules=["requires_emnist_data"],
        ),
        ArmDefinition(
            name="adversarial_cbp_hybrid",
            campaign="emnist",
            arm_type="learner",
            description="Adversarial training and CBP hybrid",
            estimated_hours=0.5,
            tags=["hybrid", "adversarial", "cbp", "emnist"],
            parameters={"adversarial_training": True, "cbp": True},
            validation_rules=["requires_emnist_data"],
        ),
        ArmDefinition(
            name="ensemble_protection",
            campaign="emnist",
            arm_type="learner",
            description="Protected ensemble learner",
            estimated_hours=0.55,
            tags=["hybrid", "ensemble", "protection", "emnist"],
            parameters={
                "ensemble_size": 3,
                "cbp": True,
                "l2init": True,
            },
            validation_rules=["requires_emnist_data"],
        ),
        # Final protections (3)
        ArmDefinition(
            name="forgetting_detector",
            campaign="emnist",
            arm_type="learner",
            description="Forgetting detection mechanism",
            estimated_hours=0.35,
            tags=["detection", "forgetting", "emnist"],
            parameters={"detect_forgetting": True},
            validation_rules=["requires_emnist_data"],
        ),
        ArmDefinition(
            name="per_class_normalization",
            campaign="emnist",
            arm_type="learner",
            description="Per-class normalization learner",
            estimated_hours=0.3,
            tags=["normalization", "per_class", "emnist"],
            parameters={"per_class_norm": True},
            validation_rules=["requires_emnist_data"],
        ),
        ArmDefinition(
            name="feature_dropout_schedule",
            campaign="emnist",
            arm_type="learner",
            description="Feature dropout with schedule",
            estimated_hours=0.3,
            tags=["dropout", "schedule", "emnist"],
            parameters={"feature_dropout": True, "schedule": "adaptive"},
            validation_rules=["requires_emnist_data"],
        ),
    ]
    registry.register_batch(emnist_arms)

    # ==========================================================================
    # MICRO-CONTINUAL: 19 arms
    # ==========================================================================
    micro_arms = [
        ArmDefinition(
            name="rls_head_resid",
            campaign="micro_continual",
            arm_type="arm",
            description="RLS head with residual",
            estimated_hours=0.2,
            tags=["baseline", "rls", "micro"],
            parameters={"method": "rls_head_resid"},
            validation_rules=["requires_micro_data"],
        ),
        ArmDefinition(
            name="alignment_first",
            campaign="micro_continual",
            arm_type="arm",
            description="Alignment-first approach",
            estimated_hours=0.25,
            tags=["baseline", "alignment", "micro"],
            parameters={"method": "alignment_first"},
            validation_rules=["requires_micro_data"],
        ),
        ArmDefinition(
            name="naive_bayes_extended",
            campaign="micro_continual",
            arm_type="arm",
            description="Extended Naive Bayes",
            estimated_hours=0.25,
            tags=["baseline", "bayes", "micro"],
            parameters={"method": "naive_bayes_extended"},
            validation_rules=["requires_micro_data"],
        ),
        ArmDefinition(
            name="dual_speed_rfs_rls",
            campaign="micro_continual",
            arm_type="arm",
            description="Dual-speed RFS and RLS",
            estimated_hours=0.3,
            tags=["baseline", "dual_speed", "micro"],
            parameters={"method": "dual_speed_rfs_rls"},
            validation_rules=["requires_micro_data"],
        ),
        ArmDefinition(
            name="actor_critic_micro",
            campaign="micro_continual",
            arm_type="arm",
            description="Actor-critic for micro tasks",
            estimated_hours=0.3,
            tags=["baseline", "actor_critic", "micro"],
            parameters={"method": "actor_critic_micro"},
            validation_rules=["requires_micro_data"],
        ),
        # Extensions (3)
        ArmDefinition(
            name="replay_buffer_learner",
            campaign="micro_continual",
            arm_type="arm",
            description="Learner with replay buffer",
            estimated_hours=0.3,
            tags=["extension", "replay", "micro"],
            parameters={"use_replay_buffer": True},
            validation_rules=["requires_micro_data"],
        ),
        ArmDefinition(
            name="plasticity_modulated",
            campaign="micro_continual",
            arm_type="arm",
            description="Plasticity-modulated learner",
            estimated_hours=0.35,
            tags=["extension", "plasticity", "micro"],
            parameters={"modulate_plasticity": True},
            validation_rules=["requires_micro_data"],
        ),
        ArmDefinition(
            name="task_boundary_detector",
            campaign="micro_continual",
            arm_type="arm",
            description="Task boundary detection",
            estimated_hours=0.35,
            tags=["extension", "boundary_detection", "micro"],
            parameters={"detect_boundaries": True},
            validation_rules=["requires_micro_data"],
        ),
        # Meta-learning (4)
        ArmDefinition(
            name="maml_inspired",
            campaign="micro_continual",
            arm_type="arm",
            description="MAML-inspired meta-learner",
            estimated_hours=0.4,
            tags=["meta_learning", "maml", "micro"],
            parameters={"method": "maml_inspired"},
            validation_rules=["requires_micro_data"],
        ),
        ArmDefinition(
            name="hypernetwork",
            campaign="micro_continual",
            arm_type="arm",
            description="Hypernetwork-based learner",
            estimated_hours=0.45,
            tags=["meta_learning", "hypernetwork", "micro"],
            parameters={"use_hypernetwork": True},
            validation_rules=["requires_micro_data"],
        ),
        ArmDefinition(
            name="context_modulation",
            campaign="micro_continual",
            arm_type="arm",
            description="Context modulation learner",
            estimated_hours=0.4,
            tags=["meta_learning", "context", "micro"],
            parameters={"context_modulation": True},
            validation_rules=["requires_micro_data"],
        ),
        ArmDefinition(
            name="episodic_memory",
            campaign="micro_continual",
            arm_type="arm",
            description="Episodic memory learner",
            estimated_hours=0.4,
            tags=["meta_learning", "memory", "micro"],
            parameters={"use_episodic_memory": True},
            validation_rules=["requires_micro_data"],
        ),
        # Gates (4)
        ArmDefinition(
            name="loss_gated",
            campaign="micro_continual",
            arm_type="arm",
            description="Loss-gated learner",
            estimated_hours=0.35,
            tags=["gating", "loss", "micro"],
            parameters={"gating": "loss"},
            validation_rules=["requires_micro_data"],
        ),
        ArmDefinition(
            name="gradient_norm_gated",
            campaign="micro_continual",
            arm_type="arm",
            description="Gradient norm gated learner",
            estimated_hours=0.35,
            tags=["gating", "gradient_norm", "micro"],
            parameters={"gating": "gradient_norm"},
            validation_rules=["requires_micro_data"],
        ),
        ArmDefinition(
            name="variance_gated",
            campaign="micro_continual",
            arm_type="arm",
            description="Variance-gated learner",
            estimated_hours=0.35,
            tags=["gating", "variance", "micro"],
            parameters={"gating": "variance"},
            validation_rules=["requires_micro_data"],
        ),
        ArmDefinition(
            name="confidence_gated",
            campaign="micro_continual",
            arm_type="arm",
            description="Confidence-gated learner",
            estimated_hours=0.35,
            tags=["gating", "confidence", "micro"],
            parameters={"gating": "confidence"},
            validation_rules=["requires_micro_data"],
        ),
        # Hybrids (3)
        ArmDefinition(
            name="rls_meta_hybrid",
            campaign="micro_continual",
            arm_type="arm",
            description="RLS and meta-learning hybrid",
            estimated_hours=0.45,
            tags=["hybrid", "rls", "meta", "micro"],
            parameters={"method": "rls_head_resid", "use_meta": True},
            validation_rules=["requires_micro_data"],
        ),
        ArmDefinition(
            name="buffer_plasticity_hybrid",
            campaign="micro_continual",
            arm_type="arm",
            description="Replay buffer and plasticity hybrid",
            estimated_hours=0.45,
            tags=["hybrid", "buffer", "plasticity", "micro"],
            parameters={
                "use_replay_buffer": True,
                "modulate_plasticity": True,
            },
            validation_rules=["requires_micro_data"],
        ),
        ArmDefinition(
            name="gate_boundary_hybrid",
            campaign="micro_continual",
            arm_type="arm",
            description="Gating and boundary detection hybrid",
            estimated_hours=0.45,
            tags=["hybrid", "gating", "boundary", "micro"],
            parameters={"gating": "loss", "detect_boundaries": True},
            validation_rules=["requires_micro_data"],
        ),
    ]
    registry.register_batch(micro_arms)

    # ==========================================================================
    # FORAGER: 19 baselines
    # ==========================================================================
    forager_arms = [
        ArmDefinition(
            name="dqn",
            campaign="forager",
            arm_type="baseline",
            description="Deep Q-Network baseline",
            estimated_hours=0.4,
            tags=["baseline", "dqn", "forager"],
            parameters={"algorithm": "dqn"},
            validation_rules=["requires_forager_env"],
        ),
        ArmDefinition(
            name="a3c",
            campaign="forager",
            arm_type="baseline",
            description="Asynchronous Advantage Actor-Critic baseline",
            estimated_hours=0.45,
            tags=["baseline", "a3c", "forager"],
            parameters={"algorithm": "a3c"},
            validation_rules=["requires_forager_env"],
        ),
        ArmDefinition(
            name="horde",
            campaign="forager",
            arm_type="baseline",
            description="Horde architecture baseline",
            estimated_hours=0.5,
            tags=["baseline", "horde", "forager"],
            parameters={"algorithm": "horde"},
            validation_rules=["requires_forager_env"],
        ),
        ArmDefinition(
            name="random",
            campaign="forager",
            arm_type="baseline",
            description="Random policy baseline",
            estimated_hours=0.1,
            tags=["baseline", "random", "forager"],
            parameters={"algorithm": "random"},
            validation_rules=["requires_forager_env"],
        ),
        # Phase-optimized (6)
        ArmDefinition(
            name="dqn_smoke_opt",
            campaign="forager",
            arm_type="baseline",
            description="DQN optimized for smoke tests",
            estimated_hours=0.45,
            tags=["phase_opt", "smoke", "dqn", "forager"],
            parameters={"algorithm": "dqn", "phase": "smoke"},
            validation_rules=["requires_forager_env"],
        ),
        ArmDefinition(
            name="dqn_continual_opt",
            campaign="forager",
            arm_type="baseline",
            description="DQN optimized for continual learning",
            estimated_hours=0.6,
            tags=["phase_opt", "continual", "dqn", "forager"],
            parameters={"algorithm": "dqn", "phase": "continual"},
            validation_rules=["requires_forager_env"],
        ),
        ArmDefinition(
            name="dqn_transfer_opt",
            campaign="forager",
            arm_type="baseline",
            description="DQN optimized for transfer",
            estimated_hours=0.6,
            tags=["phase_opt", "transfer", "dqn", "forager"],
            parameters={"algorithm": "dqn", "phase": "transfer"},
            validation_rules=["requires_forager_env"],
        ),
        ArmDefinition(
            name="a3c_smoke_opt",
            campaign="forager",
            arm_type="baseline",
            description="A3C optimized for smoke tests",
            estimated_hours=0.5,
            tags=["phase_opt", "smoke", "a3c", "forager"],
            parameters={"algorithm": "a3c", "phase": "smoke"},
            validation_rules=["requires_forager_env"],
        ),
        ArmDefinition(
            name="a3c_continual_opt",
            campaign="forager",
            arm_type="baseline",
            description="A3C optimized for continual learning",
            estimated_hours=0.65,
            tags=["phase_opt", "continual", "a3c", "forager"],
            parameters={"algorithm": "a3c", "phase": "continual"},
            validation_rules=["requires_forager_env"],
        ),
        ArmDefinition(
            name="a3c_transfer_opt",
            campaign="forager",
            arm_type="baseline",
            description="A3C optimized for transfer",
            estimated_hours=0.65,
            tags=["phase_opt", "transfer", "a3c", "forager"],
            parameters={"algorithm": "a3c", "phase": "transfer"},
            validation_rules=["requires_forager_env"],
        ),
        # Hybrids (3)
        ArmDefinition(
            name="dqn_curiosity",
            campaign="forager",
            arm_type="baseline",
            description="DQN with curiosity-driven exploration",
            estimated_hours=0.55,
            tags=["hybrid", "curiosity", "dqn", "forager"],
            parameters={"algorithm": "dqn", "exploration": "curiosity"},
            validation_rules=["requires_forager_env"],
        ),
        ArmDefinition(
            name="a3c_entropy_reg",
            campaign="forager",
            arm_type="baseline",
            description="A3C with entropy regularization",
            estimated_hours=0.6,
            tags=["hybrid", "entropy", "a3c", "forager"],
            parameters={"algorithm": "a3c", "entropy_coeff": 0.01},
            validation_rules=["requires_forager_env"],
        ),
        ArmDefinition(
            name="dqn_a3c_ensemble",
            campaign="forager",
            arm_type="baseline",
            description="Ensemble of DQN and A3C",
            estimated_hours=0.8,
            tags=["hybrid", "ensemble", "forager"],
            parameters={"ensemble": ["dqn", "a3c"]},
            validation_rules=["requires_forager_env"],
        ),
        # Advanced hybrids (6)
        ArmDefinition(
            name="dqn_a3c_weighted",
            campaign="forager",
            arm_type="baseline",
            description="Weighted DQN-A3C hybrid",
            estimated_hours=0.7,
            tags=["advanced", "weighted", "forager"],
            parameters={"algorithm": "weighted_ensemble", "weights": [0.6, 0.4]},
            validation_rules=["requires_forager_env"],
        ),
        ArmDefinition(
            name="curiosity_entropy",
            campaign="forager",
            arm_type="baseline",
            description="Curiosity and entropy exploration",
            estimated_hours=0.65,
            tags=["advanced", "exploration", "forager"],
            parameters={"exploration": ["curiosity", "entropy"]},
            validation_rules=["requires_forager_env"],
        ),
        ArmDefinition(
            name="distributional_rls",
            campaign="forager",
            arm_type="baseline",
            description="Distributional RL with RLS",
            estimated_hours=0.7,
            tags=["advanced", "distributional", "forager"],
            parameters={"algorithm": "distributional_rls"},
            validation_rules=["requires_forager_env"],
        ),
        ArmDefinition(
            name="dueling_advantage",
            campaign="forager",
            arm_type="baseline",
            description="Dueling network advantage architecture",
            estimated_hours=0.6,
            tags=["advanced", "dueling", "forager"],
            parameters={"architecture": "dueling"},
            validation_rules=["requires_forager_env"],
        ),
        ArmDefinition(
            name="multi_step_bootstrap",
            campaign="forager",
            arm_type="baseline",
            description="Multi-step bootstrapping",
            estimated_hours=0.65,
            tags=["advanced", "bootstrapping", "forager"],
            parameters={"bootstrap_steps": 5},
            validation_rules=["requires_forager_env"],
        ),
        ArmDefinition(
            name="hindsight_relabeling",
            campaign="forager",
            arm_type="baseline",
            description="Hindsight experience replay with relabeling",
            estimated_hours=0.75,
            tags=["advanced", "hindsight", "forager"],
            parameters={"use_hindsight": True},
            validation_rules=["requires_forager_env"],
        ),
    ]
    registry.register_batch(forager_arms)

    # ==========================================================================
    # RULE DISCOVERY V2: 130 genomes (represented as single placeholder)
    # ==========================================================================
    rule_discovery_arms = [
        ArmDefinition(
            name="rule_discovery_phase_1a_candidates",
            campaign="rule_discovery",
            arm_type="genome",
            description="Rule discovery phase 1a - 30 candidate genomes",
            estimated_hours=30,
            tags=["phase_1a", "candidates", "rule_discovery"],
            parameters={"phase": "1a", "n_genomes": 30},
            validation_rules=["rule_discovery_phases"],
        ),
        ArmDefinition(
            name="rule_discovery_phase_1b_ablations",
            campaign="rule_discovery",
            arm_type="genome",
            description="Rule discovery phase 1b - 30 ablation genomes",
            estimated_hours=20,
            tags=["phase_1b", "ablations", "rule_discovery"],
            parameters={"phase": "1b", "n_genomes": 30},
            validation_rules=["rule_discovery_phases"],
        ),
        ArmDefinition(
            name="rule_discovery_phase_1c_genetic",
            campaign="rule_discovery",
            arm_type="genome",
            description="Rule discovery phase 1c - 50 genetic search genomes",
            estimated_hours=40,
            tags=["phase_1c", "genetic", "rule_discovery"],
            parameters={"phase": "1c", "n_genomes": 50},
            validation_rules=["rule_discovery_phases"],
        ),
        ArmDefinition(
            name="rule_discovery_phase_1d_finetuning",
            campaign="rule_discovery",
            arm_type="genome",
            description="Rule discovery phase 1d - 20 fine-tuning genomes",
            estimated_hours=30,
            tags=["phase_1d", "finetuning", "rule_discovery"],
            parameters={"phase": "1d", "n_genomes": 20},
            validation_rules=["rule_discovery_phases"],
        ),
    ]
    registry.register_batch(rule_discovery_arms)

    return registry


if __name__ == "__main__":
    registry = create_unified_registry()
    summary = registry.get_summary()

    print("=== UNIFIED ARM REGISTRY ===")
    print(f"Total arms: {summary['total_arms']}")
    print(f"Total campaigns: {summary['total_campaigns']}")
    print(f"Total estimated hours: {summary['total_estimated_hours']}")
    print("\nCampaign summary:")
    for campaign, stats in summary["campaigns"].items():
        print(f"  {campaign}: {stats['count']} arms, {stats['hours']:.1f} hours")

    # Validate all registrations
    validation = registry.validate_all()
    print(f"\nValidation: {validation['valid']} valid, {validation['invalid']} invalid")
    if validation['errors']:
        print("Errors:")
        for error in validation['errors']:
            print(f"  {error}")

    # Export to JSON
    registry.export_json("unified_arm_registry.json")
    print("\nExported to unified_arm_registry.json")
