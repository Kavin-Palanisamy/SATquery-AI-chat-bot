from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from satquery.schemas.evidence import EvidenceItem
from satquery.schemas.confidence import ConfidenceRecord


class ExecutionStep(BaseModel):
    """Step in the observable multi-step execution plan."""
    step_id: int = Field(..., description="Step sequence index (1-based)")
    name: str = Field(..., description="Step identifier (e.g., 'InputValidation', 'QueryIntentAnalysis', 'SpecialistExecution')")
    description: str = Field(..., description="Human-readable description of the operation executed in this step")
    tool: Optional[str] = Field(None, description="Tool invoked for this step")
    model: Optional[str] = Field(None, description="Specialist model utilized")
    provenance: Optional[str] = Field(None, description="Provenance of model/tool in this step")
    status: str = Field("SUCCESS", description="Step execution status: SUCCESS, FAILED, SKIPPED")
    latency_sec: float = Field(0.0, description="Execution time of this step in seconds")
    details: Dict[str, Any] = Field(default_factory=dict, description="Step outputs and telemetry")


class ExecutionTrace(BaseModel):
    """Full auditable operational execution trace."""
    request_id: str = Field(..., description="Unique execution UUID")
    timestamp: str = Field(..., description="ISO-8601 execution timestamp")
    input_summary: Dict[str, Any] = Field(default_factory=dict, description="Summary of validated input files and modalities")
    query: str = Field(..., description="Original user prompt")
    intent: Dict[str, Any] = Field(default_factory=dict, description="Parsed query intent and requirements")
    plan: List[str] = Field(default_factory=list, description="Planned execution pipeline stages")
    steps: List[ExecutionStep] = Field(default_factory=list, description="Step-by-step execution details")
    models_used: List[Dict[str, Any]] = Field(default_factory=list, description="List of models with provenance invoked")
    tools_used: List[str] = Field(default_factory=list, description="List of specialist tools invoked")
    evidence: List[EvidenceItem] = Field(default_factory=list, description="All verified evidence items collected")
    confidence: Optional[ConfidenceRecord] = Field(None, description="Structured overall confidence evaluation")
    warnings: List[str] = Field(default_factory=list, description="Warnings or model limitations encountered")
    errors: List[str] = Field(default_factory=list, description="Non-fatal or fatal error messages")
    total_latency_sec: float = Field(0.0, description="Total end-to-end execution time in seconds")
    final_status: str = Field("COMPLETED", description="Final execution outcome: COMPLETED, REJECTED, FAILED")
