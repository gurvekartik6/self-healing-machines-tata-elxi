"""Optional Gemini + LangGraph maintenance investigation workflow.

The graph summarizes already-computed evidence. It never changes RUL predictions,
triage priorities, model approval, or creates work orders by itself.
"""
from __future__ import annotations
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict, Any

from src.config import REPORTS

class MaintenanceState(TypedDict, total=False):
    engine_id: int
    evidence: dict[str, Any]
    prompt: str
    retrieved_docs: list[dict[str, Any]]
    analysis: dict[str, Any]
    error: str


def _evidence_node(state: MaintenanceState) -> MaintenanceState:
    """Build bounded, numeric evidence from the latest saved predictions and rules."""
    pred_path = REPORTS / "predictions.csv"
    decision_path = REPORTS / "triage_decisions.csv"
    if not pred_path.exists():
        raise FileNotFoundError("Run the predictive-maintenance pipeline first.")
    import pandas as pd
    predictions = pd.read_csv(pred_path)
    matches = predictions[predictions["engine_id"].astype(int) == int(state["engine_id"])]
    if matches.empty:
        raise ValueError(f"Engine {state['engine_id']} was not found in the latest predictions.")
    row = matches.iloc[0]
    decision = {}
    if decision_path.exists():
        d = pd.read_csv(decision_path)
        d = d[d.engine_id.astype(int) == int(state["engine_id"])]
        if not d.empty:
            decision = {"rule_decision": str(d.iloc[-1].get("decision", "UNKNOWN")),
                        "rule_reason": str(d.iloc[-1].get("reason", ""))}
    sensors = {k: float(row[k]) for k in row.index if str(k).startswith("sensor_") and pd.notna(row[k])}
    evidence = {
        "engine_id": int(row["engine_id"]),
        "cycle": int(row["cycle"]),
        "predicted_rul_cycles": round(float(row["predicted_rul"]), 2),
        "anomaly_flag": bool(row["anomaly"]),
        "sensor_features": sensors,
        "decision": decision,
        "evidence_limits": [
            "Predicted RUL is an estimate, not a guarantee or measured failure time.",
            "IsolationForest anomaly flag means unusual relative to its training distribution; it does not prove a fault.",
            "FD001 is simulated turbofan data and may not transfer to other equipment.",
        ],
    }
    prompt = (
        "You are a maintenance decision-support assistant. Use only the supplied evidence. "
        "Do not invent causes, sensor units, thresholds, parts, or maintenance procedures. "
        "If evidence is insufficient, say so and recommend qualified human inspection. "
        "Do not override the rule decision or authorize repair. Return valid JSON with keys: "
        "summary (string), evidence (array of strings), possible_causes (array of cautious hypotheses), "
        "recommended_checks (array of safe non-invasive checks), uncertainty (string), "
        "human_review_required (boolean). State that hypotheses are not confirmed diagnoses.\n\n"
        f"Evidence JSON: {json.dumps(evidence, ensure_ascii=False)}"
    )
    return {"evidence": evidence, "prompt": prompt}


def _retrieval_node(state: MaintenanceState) -> MaintenanceState:
    """Retrieve local maintenance references; empty retrieval is a valid outcome."""
    from src.maintenance_rag import retrieve
    evidence = state["evidence"]
    decision = evidence.get("decision", {})
    query = (
        f"Engine maintenance investigation rule decision {decision.get('rule_decision', 'UNKNOWN')} "
        f"reason {decision.get('rule_reason', '')} anomaly {evidence.get('anomaly_flag')} "
        f"predicted remaining useful life {evidence.get('predicted_rul_cycles')} cycles "
        "sensor trend inspection troubleshooting vibration temperature pressure"
    )
    docs = retrieve(query, top_k=4, asset_model=os.getenv("TELIPORT_ASSET_MODEL", "FD001"))
    return {"retrieved_docs": docs}


def _gemini_node(state: MaintenanceState) -> MaintenanceState:
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Set GOOGLE_API_KEY (or GEMINI_API_KEY) to enable Gemini analysis.")
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.messages import HumanMessage
    except ImportError as exc:
        raise RuntimeError("Install optional LLM dependencies: pip install -r requirements-llm.txt") from exc
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    llm = ChatGoogleGenerativeAI(model=model_name, temperature=0, google_api_key=api_key)
    retrieved = state.get("retrieved_docs", [])
    if retrieved:
        reference_context = "\n".join(
            f"[SOURCE {i + 1}] file={doc['source']} title={doc.get('title','')} revision={doc.get('revision','')} page={doc.get('page')} section={doc.get('section')} chunk={doc['chunk']} score={doc['score']} approved_by={doc.get('approved_by','')}: {doc['text']}"
            for i, doc in enumerate(retrieved)
        )
        retrieval_note = "Use the reference excerpts only as supporting context. Cite only the supplied source filenames. Do not treat a retrieved excerpt as proof that a fault exists."
    else:
        reference_context = "NO MAINTENANCE REFERENCES WERE RETRIEVED. Do not invent manual citations or equipment-specific procedures. State that no approved reference was available and limit recommendations to general, non-invasive checks plus qualified human review."
        retrieval_note = "No retrieved reference means no manual-backed procedure can be recommended."
    prompt = state["prompt"] + "\n\nRetrieved maintenance references:\n" + reference_context + "\n\n" + retrieval_note + "\nReturn one additional key, source_citations, as an array of objects with source and relevance. Use an empty array when no sources were retrieved."
    # Bounded retry with exponential backoff for transient API/quota/network failures.
    response = None
    last_error = None
    for attempt in range(3):
        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            break
        except Exception as exc:
            last_error = exc
            if attempt == 2:
                break
            time.sleep(2 ** attempt)
    if response is None:
        raise RuntimeError(f"Gemini request failed after 3 attempts; no action was taken. Check API key, model access, quota and network. Details: {type(last_error).__name__}: {last_error}")
    content = response.content
    if isinstance(content, list):
        content = "".join(str(x.get("text", "")) if isinstance(x, dict) else str(x) for x in content)
    content = str(content).strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        analysis = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError("Gemini returned invalid JSON. Retry the analysis; no action was taken.") from exc
    required = {"summary", "evidence", "possible_causes", "recommended_checks", "uncertainty", "human_review_required", "source_citations"}
    if not required.issubset(analysis):
        raise ValueError("Gemini response is missing required fields. No action was taken.")
    # Force the safety boundary regardless of model output.
    analysis["human_review_required"] = True
    # Citations are constrained to sources actually retrieved by the local retriever.
    allowed_sources = {doc["source"] for doc in retrieved}
    citations = analysis.get("source_citations", [])
    if not isinstance(citations, list):
        citations = []
    analysis["source_citations"] = [c for c in citations if isinstance(c, dict) and c.get("source") in allowed_sources]
    # Add canonical source/page/section provenance so users need not trust model-invented locations.
    for citation in analysis["source_citations"]:
        matching = next((d for d in retrieved if d["source"] == citation.get("source")), None)
        if matching:
            citation["title"] = matching.get("title")
            citation["revision"] = matching.get("revision")
            citation["page"] = matching.get("page")
            citation["section"] = matching.get("section")
    analysis["model"] = model_name
    return {"analysis": analysis}


def build_graph():
    try:
        from langgraph.graph import StateGraph, START, END
    except ImportError as exc:
        raise RuntimeError("Install optional LLM dependencies: pip install -r requirements-llm.txt") from exc
    graph = StateGraph(MaintenanceState)
    graph.add_node("collect_evidence", _evidence_node)
    graph.add_node("retrieve_maintenance_refs", _retrieval_node)
    graph.add_node("gemini_analysis", _gemini_node)
    graph.add_edge(START, "collect_evidence")
    graph.add_edge("collect_evidence", "retrieve_maintenance_refs")
    graph.add_edge("retrieve_maintenance_refs", "gemini_analysis")
    graph.add_edge("gemini_analysis", END)
    return graph.compile()


def analyze_engine(engine_id: int) -> dict[str, Any]:
    """Run one LangGraph analysis and persist a timestamped report."""
    app = build_graph()
    result = app.invoke({"engine_id": int(engine_id)})
    output = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "engine_id": int(engine_id),
        "evidence": result["evidence"],
        "retrieved_references": result.get("retrieved_docs", []),
        "analysis": result["analysis"],
        "notice": "Decision support only. No work order was created or modified by the LLM.",
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    path = REPORTS / f"llm_analysis_engine_{int(engine_id)}.json"
    path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    from src.investigation_store import save_investigation
    save_investigation(int(engine_id), str(path.relative_to(Path(__file__).resolve().parents[1])), output)
    return output
