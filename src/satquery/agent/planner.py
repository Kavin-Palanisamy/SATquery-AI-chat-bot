from typing import List
from pydantic import BaseModel, Field
from satquery.schemas.intent import AgentTaskType, QueryIntent


class ExecutionPlan(BaseModel):
    """Declarative execution plan detailing pipeline stages and tool invocations."""
    plan_id: str = Field(..., description="Plan identifier (PLAN_A, PLAN_B, PLAN_C, PLAN_D, PLAN_E, PLAN_F)")
    name: str = Field(..., description="Human-readable workflow title")
    stages: List[str] = Field(default_factory=list, description="Ordered execution pipeline stages")
    primary_tool: str = Field(..., description="Main tool responsible for primary inference")
    auxiliary_tools: List[str] = Field(default_factory=list, description="Supporting tools invoked for evidence generation")
    requires_dual_input: bool = Field(False, description="True if workflow mandates two input images")


def create_execution_plan(intent: QueryIntent, num_images: int = 1) -> ExecutionPlan:
    """
    Constructs a multi-step execution plan based on query intent and input resources.
    """
    task = intent.task

    if task == AgentTaskType.OPTICAL_SAR_FUSION:
        return ExecutionPlan(
            plan_id="PLAN_D_OPTICAL_SAR_FUSION",
            name="Cross-Modal Optical + SAR Multi-Sensor Fusion Pipeline",
            stages=[
                "InputValidation",
                "ModalityVerification",
                "SpatialCoregistrationCheck",
                "CrossModalFeatureExtraction",
                "EvidenceSynthesis",
                "AuditLogging",
            ],
            primary_tool="OpticalSARFusionTool",
            auxiliary_tools=["InputValidationTool", "RemoteSensingVQATool"],
            requires_dual_input=True,
        )

    if task == AgentTaskType.BITEMPORAL_CHANGE:
        return ExecutionPlan(
            plan_id="PLAN_C_CHANGE_ANALYSIS",
            name="Bi-Temporal Change Vector Analysis & CDVQA Pipeline",
            stages=[
                "InputValidation",
                "TemporalPairDimensionResampling",
                "SpectralDifferencing",
                "ChangeMaskGeneration",
                "ChangeVQASynthesis",
                "AuditLogging",
            ],
            primary_tool="RemoteSensingChangeTool",
            auxiliary_tools=["InputValidationTool", "RemoteSensingVQATool"],
            requires_dual_input=True,
        )

    if task == AgentTaskType.SEGMENTATION_AREA:
        return ExecutionPlan(
            plan_id="PLAN_B_SEGMENTATION_AREA",
            name="Quantitative Land-Cover Pixel Segmentation & Coverage Pipeline",
            stages=[
                "InputValidation",
                "QueryIntentAnalysis",
                "PixelSegmentation",
                "AreaMetricCalculation",
                "EvidenceAwareVQASynthesis",
                "AuditLogging",
            ],
            primary_tool="SegmentationTool",
            auxiliary_tools=["InputValidationTool", "RemoteSensingVQATool"],
            requires_dual_input=False,
        )

    if task == AgentTaskType.DETECTION_COUNT:
        return ExecutionPlan(
            plan_id="PLAN_E_DETECTION_COUNT",
            name="Object Instance Detection & Discrete Counting Pipeline",
            stages=[
                "InputValidation",
                "QueryIntentAnalysis",
                "MorphologicalDetection",
                "InstanceCounting",
                "EvidenceAwareVQASynthesis",
                "AuditLogging",
            ],
            primary_tool="DetectionTool",
            auxiliary_tools=["InputValidationTool", "RemoteSensingVQATool"],
            requires_dual_input=False,
        )

    if task == AgentTaskType.GROUNDING:
        return ExecutionPlan(
            plan_id="PLAN_F_DIRECT_GROUNDING",
            name="Spatial Grounding & Natural Language Localization Pipeline",
            stages=[
                "InputValidation",
                "QueryIntentAnalysis",
                "SpectralHeuristicGrounding",
                "CoordinateValidation",
                "MaskExtraction",
                "AuditLogging",
            ],
            primary_tool="RemoteSensingGroundingTool",
            auxiliary_tools=["InputValidationTool"],
            requires_dual_input=False,
        )

    # Default VQA / Scene Description
    return ExecutionPlan(
        plan_id="PLAN_A_VQA_GROUNDING",
        name="Multimodal VQA & Qualitative Scene Description Pipeline",
        stages=[
            "InputValidation",
            "QueryIntentAnalysis",
            "SpecialistVQAExecution",
            "LimitationVerification",
            "AnswerSynthesis",
            "AuditLogging",
        ],
        primary_tool="RemoteSensingVQATool",
        auxiliary_tools=["InputValidationTool", "RemoteSensingGroundingTool"],
        requires_dual_input=False,
    )
