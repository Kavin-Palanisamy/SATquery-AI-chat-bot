import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from PIL import Image


from satquery.agent.intent import parse_query_intent
from satquery.agent.planner import create_execution_plan
from satquery.agent.synthesizer import AnswerSynthesizer
from satquery.geo.validator import PairValidationResult, ValidationResult, validate_image_pair, validate_input_image
from satquery.models.registry import get_model_registry
from satquery.schemas.confidence import CalibrationStatus, ConfidenceRecord
from satquery.schemas.evidence import EvidenceItem, EvidenceType
from satquery.schemas.execution import ExecutionStep, ExecutionTrace
from satquery.schemas.intent import AgentTaskType, QueryIntent
from satquery.schemas.vqa import AgentResponse, BoundingBox
from satquery.tools.registry import get_tool_registry
from satquery.utils.logging import get_logger

logger = get_logger("satquery.agent.executor")


class AgentExecutor:
    """
    Agentic Orchestrator Executor V3.
    Coordinates input validation, intent parsing, multi-step tool execution, and observable trace logging.
    """

    def __init__(self):
        self.tool_registry = get_tool_registry()
        self.model_registry = get_model_registry()

    def execute(
        self,
        query: str,
        primary_image: Union[str, Image.Image],
        secondary_image: Optional[Union[str, Image.Image]] = None,
        task_override: Optional[str] = None,
        primary_metadata: Optional[Dict[str, Any]] = None,
        secondary_metadata: Optional[Dict[str, Any]] = None,
    ) -> AgentResponse:
        start_time = time.time()
        req_id = str(uuid.uuid4())
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        trace_steps: List[ExecutionStep] = []
        warnings: List[str] = []
        errors: List[str] = []
        evidence_items: List[EvidenceItem] = []
        limitations: List[str] = []

        # Convert images to PIL
        if isinstance(primary_image, (str, Path)):
            from satquery.geo.image_loader import load_image
            geo1 = load_image(primary_image)
            img1 = geo1.pil_image
            primary_metadata = primary_metadata or geo1.metadata
        else:
            img1 = primary_image
            primary_metadata = primary_metadata or {}

        img2 = None
        if secondary_image is not None:
            if isinstance(secondary_image, (str, Path)):
                from satquery.geo.image_loader import load_image
                geo2 = load_image(secondary_image)
                img2 = geo2.pil_image
                secondary_metadata = secondary_metadata or geo2.metadata
            else:
                img2 = secondary_image
                secondary_metadata = secondary_metadata or {}

        num_images = 2 if img2 is not None else 1

        # -------------------------------------------------------------
        # STEP 1: Input Validation
        # -------------------------------------------------------------
        s1_start = time.time()
        val_tool = self.tool_registry.get_tool("InputValidationTool")
        val1, _ = validate_input_image(img1)
        trace_steps.append(
            ExecutionStep(
                step_id=1,
                name="InputValidation",
                description=f"Validated primary image ({val1.shape[1]}x{val1.shape[0]} px, {val1.modality}, {val1.bands} bands).",
                tool="InputValidationTool",
                status="SUCCESS" if val1.valid else "FAILED",
                latency_sec=round(time.time() - s1_start, 4),
                details={"primary": val1.model_dump()},
            )
        )

        modalities = [val1.modality]
        if img2 is not None:
            val2, _ = validate_input_image(img2)
            modalities.append(val2.modality)

        # -------------------------------------------------------------
        # STEP 2: Query Intent Analysis
        # -------------------------------------------------------------
        s2_start = time.time()
        intent = parse_query_intent(query, num_images=num_images, modalities=modalities)
        if task_override and task_override.lower() != "auto":
            try:
                intent.task = AgentTaskType(task_override.lower())
            except ValueError:
                pass

        trace_steps.append(
            ExecutionStep(
                step_id=2,
                name="QueryIntentAnalysis",
                description=f"Classified intent as '{intent.intent}' for task '{intent.task.value}'.",
                status="SUCCESS",
                latency_sec=round(time.time() - s2_start, 4),
                details=intent.model_dump(),
            )
        )

        # -------------------------------------------------------------
        # STEP 3: Task Planning
        # -------------------------------------------------------------
        s3_start = time.time()
        plan = create_execution_plan(intent, num_images=num_images)
        trace_steps.append(
            ExecutionStep(
                step_id=3,
                name="TaskPlanning",
                description=f"Generated execution plan '{plan.plan_id}': {plan.name}.",
                status="SUCCESS",
                latency_sec=round(time.time() - s3_start, 4),
                details=plan.model_dump(),
            )
        )


        # -------------------------------------------------------------
        # VALIDATION GATE: Dual-image check
        # -------------------------------------------------------------
        if plan.requires_dual_input and img2 is None:
            msg = (
                f"Workflow '{intent.task.value}' requires two uploaded images, but only one image was provided. "
                "Please upload both primary and secondary scenes to execute this analysis."
            )
            total_lat = time.time() - start_time
            trace = ExecutionTrace(
                request_id=req_id,
                timestamp=timestamp,
                input_summary={"num_images": 1, "modalities": modalities},
                query=query,
                intent=intent.model_dump(),
                plan=plan.stages,
                steps=trace_steps,
                tools_used=["InputValidationTool"],
                errors=[msg],
                total_latency_sec=round(total_lat, 3),
                final_status="REJECTED",
            )
            return AnswerSynthesizer.synthesize(
                task=AgentTaskType.VALIDATION_ERROR.value,
                answer=msg,
                tool_used="InputValidationTool",
                model_name="ValidatorEngine",
                model_mode="REAL",
                rs_adaptation="NOT LOADED",
                confidence=None,
                execution_time_sec=total_lat,
                execution_trace=trace.model_dump(),
            )

        if intent.task == AgentTaskType.OPTICAL_SAR_FUSION and img2 is not None:
            mods = set(modalities)
            if "OPTICAL" not in mods or "SAR" not in mods:
                msg = (
                    f"Optical-SAR fusion requires one Optical and one SAR radar scene. "
                    f"Received modalities: {modalities[0]} and {modalities[1]}."
                )
                total_lat = time.time() - start_time
                trace = ExecutionTrace(
                    request_id=req_id,
                    timestamp=timestamp,
                    input_summary={"num_images": 2, "modalities": modalities},
                    query=query,
                    intent=intent.model_dump(),
                    plan=plan.stages,
                    steps=trace_steps,
                    tools_used=["InputValidationTool"],
                    errors=[msg],
                    total_latency_sec=round(total_lat, 3),
                    final_status="REJECTED",
                )
                return AnswerSynthesizer.synthesize(
                    task=AgentTaskType.VALIDATION_ERROR.value,
                    answer=msg,
                    tool_used="InputValidationTool",
                    model_name="ValidatorEngine",
                    model_mode="REAL",
                    rs_adaptation="NOT LOADED",
                    confidence=None,
                    execution_time_sec=total_lat,
                    execution_trace=trace.model_dump(),
                )

        # -------------------------------------------------------------
        # STEP 4: Model Selection & Specialist Execution
        # -------------------------------------------------------------
        s4_start = time.time()
        final_answer = ""
        tool_used_name = plan.primary_tool
        model_name = "Qwen/Qwen2-VL-2B-Instruct"
        model_mode = "MOCK"
        rs_adaptation = "NOT LOADED"
        confidence_val: Optional[float] = None
        evidence_used_label = "NONE"
        boxes_out: Optional[List[BoundingBox]] = None
        change_map_url_out: Optional[str] = None
        grounding_mask_url_out: Optional[str] = None
        grounding_res_out: Optional[Dict[str, Any]] = None

        # ROUTE 1: Grounding Workflow
        if intent.task == AgentTaskType.GROUNDING:
            ground_tool = self.tool_registry.get_tool("RemoteSensingGroundingTool")
            gr = ground_tool.execute(img1, query, target=intent.target)
            model_name = "RS-SpectralHeuristic-Grounder-v1"
            model_mode = "HEURISTIC"
            rs_adaptation = "NOT LOADED"
            confidence_val = gr.confidence

            if gr.success and gr.detections:
                evidence_used_label = "GROUNDING"
                d = gr.detections[0]
                boxes_out = [
                    BoundingBox(
                        ymin=d.ymin,
                        xmin=d.xmin,
                        ymax=d.ymax,
                        xmax=d.xmax,
                        label=d.label.title(),
                        confidence=d.confidence,
                        bbox_coverage_pct=d.bbox_coverage_pct,
                        mask_coverage_pct=d.mask_coverage_pct,
                        confidence_type="heuristic",
                    )
                ]
                grounding_mask_url_out = gr.mask_url
                grounding_res_out = gr.model_dump()
                evidence_items.append(

                    EvidenceItem(
                        type=EvidenceType.BBOX,
                        source=model_name,
                        target=gr.target,
                        bbox={"ymin": d.ymin, "xmin": d.xmin, "ymax": d.ymax, "xmax": d.xmax},
                        bbox_coverage_pct=d.bbox_coverage_pct,
                        mask_coverage_pct=d.mask_coverage_pct,
                        confidence=d.confidence,
                        description=f"Localized {gr.target} candidate boundary",
                        provenance="HEURISTIC",
                    )
                )
                final_answer = (
                    f"Localized primary {gr.target} boundary spanning "
                    f"[{int(d.xmin*100)}%–{int(d.xmax*100)}% X, {int(d.ymin*100)}%–{int(d.ymax*100)}% Y]. "
                    f"Bounding box covers {d.bbox_coverage_pct:.1f}% of the image. "
                    f"Detected mask covers {d.mask_coverage_pct or 0.0:.2f}% of the image (heuristic grounding)."
                )
            else:
                final_answer = f"Unable to reliably localize the requested target '{gr.target}' using the available heuristic grounding method."

        # ROUTE 2: Segmentation Area Workflow
        elif intent.task == AgentTaskType.SEGMENTATION_AREA:
            seg_tool = self.tool_registry.get_tool("SegmentationTool")
            sr = seg_tool.execute(img1, target_class=intent.target or "water")
            vqa_tool = self.tool_registry.get_tool("RemoteSensingVQATool")

            if sr.success:
                evidence_used_label = "SEGMENTATION"
                grounding_mask_url_out = sr.mask_url
                evidence_items.append(
                    EvidenceItem(
                        type=EvidenceType.MASK,
                        source="RS-SpectralHeuristic-Segmenter-v1",
                        target=sr.target_class,
                        mask_coverage_pct=sr.area_percent,
                        area_pixels=sr.positive_pixels,
                        valid_pixels=sr.valid_pixels,
                        confidence=sr.confidence,
                        description=f"Measured {sr.area_percent:.1f}% pixel coverage for {sr.target_class}",
                        provenance="HEURISTIC",
                    )
                )
                vqa_res = vqa_tool.execute(img1, query, context={"segmentation": sr.model_dump()})
                final_answer = vqa_res.answer
                model_name = vqa_res.model_info.name
                model_mode = vqa_res.model_info.mode
                rs_adaptation = vqa_res.model_info.rs_adaptation_status
                confidence_val = None
            else:
                limitations.append("Segmentation mask could not be extracted.")
                vqa_res = vqa_tool.execute(img1, query, context={})
                final_answer = vqa_res.answer
                model_name = vqa_res.model_info.name
                model_mode = vqa_res.model_info.mode
                rs_adaptation = vqa_res.model_info.rs_adaptation_status
                confidence_val = None

        # ROUTE 3: Detection Count Workflow
        elif intent.task == AgentTaskType.DETECTION_COUNT:
            det_tool = self.tool_registry.get_tool("DetectionTool")
            dr = det_tool.execute(img1, target_class=intent.target or "building")
            vqa_tool = self.tool_registry.get_tool("RemoteSensingVQATool")

            if dr.success and dr.total_count > 0:
                evidence_used_label = "DETECTION"
                boxes_out = [
                    BoundingBox(
                        ymin=obj.bbox.ymin,
                        xmin=obj.bbox.xmin,
                        ymax=obj.bbox.ymax,
                        xmax=obj.bbox.xmax,
                        label=obj.label.title(),
                        confidence=obj.confidence,
                        bbox_coverage_pct=obj.bbox.bbox_coverage_pct,
                        confidence_type="heuristic",
                    )
                    for obj in dr.objects[:10]  # Cap visual boxes to top 10
                ]
                for obj in dr.objects:
                    evidence_items.append(
                        EvidenceItem(
                            type=EvidenceType.DETECTION,
                            source="RS-Morphological-Detector-v1",
                            target=obj.label,
                            bbox={"ymin": obj.bbox.ymin, "xmin": obj.bbox.xmin, "ymax": obj.bbox.ymax, "xmax": obj.bbox.xmax},
                            bbox_coverage_pct=obj.bbox.bbox_coverage_pct,
                            confidence=obj.confidence,
                            description=f"Detected {obj.label} instance",
                            provenance="HEURISTIC",
                        )
                    )
                vqa_res = vqa_tool.execute(img1, query, context={"detection": dr.model_dump()})
                final_answer = vqa_res.answer
                model_name = vqa_res.model_info.name
                model_mode = vqa_res.model_info.mode
                rs_adaptation = vqa_res.model_info.rs_adaptation_status
                confidence_val = None
            else:
                vqa_res = vqa_tool.execute(img1, query, context={})
                final_answer = vqa_res.answer
                model_name = vqa_res.model_info.name
                model_mode = vqa_res.model_info.mode
                rs_adaptation = vqa_res.model_info.rs_adaptation_status
                confidence_val = None

        # ROUTE 4: Bi-Temporal Change Workflow
        elif intent.task == AgentTaskType.BITEMPORAL_CHANGE and img2 is not None:
            change_tool = self.tool_registry.get_tool("RemoteSensingChangeTool")
            cr = change_tool.execute(img1, img2, query, meta_t1=primary_metadata, meta_t2=secondary_metadata)
            final_answer = cr.summary
            change_map_url_out = cr.change_map_url
            confidence_val = cr.confidence
            model_mode = "real"
            evidence_used_label = "NONE"
            if cr.changed:
                evidence_items.append(
                    EvidenceItem(
                        type=EvidenceType.CHANGE_MASK,
                        source="RS-ClassicalSpectralChange-v1",
                        target="surface_alteration",
                        mask_coverage_pct=cr.change_area_percent,
                        confidence=cr.confidence,
                        description=f"Detected {cr.change_area_percent:.1f}% spectral difference between T1 and T2",
                        provenance="REAL",
                    )
                )

        # ROUTE 5: Optical + SAR Fusion Workflow
        elif intent.task == AgentTaskType.OPTICAL_SAR_FUSION and img2 is not None:
            fusion_tool = self.tool_registry.get_tool("OpticalSARFusionTool")
            fr = fusion_tool.execute(img1, img2, query, optical_meta=primary_metadata, sar_meta=secondary_metadata)
            final_answer = fr.summary
            confidence_val = fr.confidence
            model_mode = "real"
            evidence_used_label = "NONE"
            for ev_str in fr.fused_evidence:
                evidence_items.append(
                    EvidenceItem(
                        type=EvidenceType.CROSS_MODAL,
                        source="RS-CrossModalSynergy-v1",
                        target="crossmodal_synergy",
                        description=ev_str,
                        provenance="REAL",
                    )
                )

        # ROUTE 6: Standard VQA / Scene Description
        else:
            vqa_tool = self.tool_registry.get_tool("RemoteSensingVQATool")
            vqa_res = vqa_tool.execute(img1, query, context={})
            final_answer = vqa_res.answer
            model_name = vqa_res.model_info.name
            model_mode = vqa_res.model_info.mode
            rs_adaptation = vqa_res.model_info.rs_adaptation_status
            confidence_val = None

        trace_steps.append(
            ExecutionStep(
                step_id=4,
                name="SpecialistExecution",
                description=f"Executed specialist tool '{tool_used_name}' using model '{model_name}' (Mode: {model_mode}).",
                tool=tool_used_name,
                model=model_name,
                provenance=model_mode,
                status="SUCCESS",
                latency_sec=round(time.time() - s4_start, 4),
            )
        )

        # -------------------------------------------------------------
        # STEP 5: Evidence Validation & Synthesis
        # -------------------------------------------------------------
        total_lat = time.time() - start_time
        
        # Build backward-compatible trace dictionary matching existing UI/test assertions
        trace_dict = {
            "task": intent.task.value,
            "decision_reasoning": intent.reason,
            "routing_reason": intent.reason,
            "task_reason": intent.reason,
            "intent": intent.model_dump(),
            "intent_reason": intent.reason,
            "confidence_breakdown": {
                "routing_confidence": intent.confidence,
                "model_confidence": confidence_val,
                "evidence_confidence": confidence_val if evidence_items else None,
                "is_mock": model_mode.upper() == "MOCK",
            },
            "steps": [s.model_dump() for s in trace_steps],
            "evidence_collected": [e.model_dump() for e in evidence_items],
        }


        # Unified task name for existing API schemas
        task_str = intent.task.value
        if task_str in ("scene_description", "segmentation_area", "detection_count"):
            task_str = "vqa"

        return AnswerSynthesizer.synthesize(
            task=task_str,
            answer=final_answer,
            tool_used=tool_used_name,
            model_name=model_name,
            model_mode=model_mode,
            rs_adaptation=rs_adaptation,
            confidence=confidence_val,
            evidence_used=evidence_used_label,
            evidence_items=evidence_items,
            boxes=boxes_out,
            change_map_url=change_map_url_out,
            grounding_mask_url=grounding_mask_url_out,
            grounding_result=grounding_res_out,
            metadata=primary_metadata,
            execution_time_sec=total_lat,
            execution_trace=trace_dict,
            limitations=limitations,
        )
