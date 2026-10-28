
"""TELIPORT Self-Healing Machines — user-friendly Streamlit dashboard."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import REPORTS
from src.triage import (
    read_orders,
    update_status,
    assign_order,
    acknowledge_order,
    escalate_order,
    read_audit,
)
from src.lifecycle import approve_candidate
from dotenv import load_dotenv

# Load .env from the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# Read Gemini configuration
GEMINI_API_KEY = (
    os.getenv("GOOGLE_API_KEY")
    or os.getenv("GEMINI_API_KEY")
)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


# ---------------------------------------------------------
# PAGE CONFIGURATION AND STYLING
# ---------------------------------------------------------

st.set_page_config(
    page_title="TELIPORT | Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1450px;
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
    }

    [data-testid="stSidebar"] {
        border-right: 1px solid rgba(128,128,128,.22);
    }

    .hero {
        padding: 1.4rem 1.6rem;
        border: 1px solid rgba(128,128,128,.25);
        border-radius: 16px;
        margin-bottom: 1.2rem;
    }

    .hero h1 {
        font-size: 1.9rem;
        margin-bottom: .35rem;
    }

    .hero p {
        opacity: .75;
        margin-bottom: 0;
    }

    .section-note {
        opacity: .75;
        margin-top: -.35rem;
        margin-bottom: 1rem;
    }

    div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,.20);
        border-radius: 12px;
        padding: 12px 14px;
    }

    .workflow-step {
        padding: .8rem 1rem;
        border: 1px solid rgba(128,128,128,.20);
        border-radius: 10px;
        margin-bottom: .5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# OPTIONAL PASSWORD GATE
# ---------------------------------------------------------

expected_password = os.getenv("TELIPORT_DASHBOARD_PASSWORD")

if expected_password:
    st.sidebar.markdown("### 🔐 Secure access")
    provided_password = st.sidebar.text_input(
        "Dashboard password",
        type="password",
    )

    if provided_password != expected_password:
        st.title("TELIPORT")
        st.info("Enter the dashboard password in the sidebar to continue.")
        st.stop()


# ---------------------------------------------------------
# PATHS AND DATA LOADING
# ---------------------------------------------------------

metrics_path = REPORTS / "metrics.json"
predictions_path = REPORTS / "predictions.csv"
decisions_path = REPORTS / "triage_decisions.csv"


@st.cache_data(show_spinner=False)
def load_json(path_text, modified_time):
    path = Path(path_text)

    if not path.exists():
        return {}

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


@st.cache_data(show_spinner=False)
def load_csv(path_text, modified_time):
    path = Path(path_text)

    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError):
        return pd.DataFrame()


def file_version(path):
    return path.stat().st_mtime if path.exists() else 0


def refresh_data():
    st.cache_data.clear()
    st.rerun()


metrics = load_json(str(metrics_path), file_version(metrics_path))
predictions = load_csv(
    str(predictions_path),
    file_version(predictions_path),
)

try:
    orders = read_orders()
except Exception:
    orders = pd.DataFrame()


# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------

def fmt(value, digits=1):
    try:
        number = float(value)

        if pd.isna(number):
            return "—"

        return f"{number:,.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def is_flagged(value):
    if isinstance(value, bool):
        return value

    return str(value).strip().lower() in {
        "true", "1", "yes", "flagged", "anomaly"
    }


def get_decision(row):
    return row.get(
        "decision",
        row.get("triage", "MONITOR"),
    )


def get_rul(row):
    return row.get("predicted_rul", None)


def show_table(dataframe, rename=None):
    if dataframe is None or dataframe.empty:
        st.info("No records are available yet.")
        return

    view = dataframe.copy()

    if rename:
        view = view.rename(
            columns={
                key: value
                for key, value in rename.items()
                if key in view.columns
            }
        )

    st.dataframe(
        view,
        width="stretch",
        hide_index=True,
    )


def show_prediction_table(dataframe):
    show_table(
        dataframe,
        {
            "engine_id": "Engine",
            "predicted_rul": "Predicted RUL (cycles)",
            "anomaly": "Unusual pattern",
            "decision": "Priority",
            "triage": "Priority",
        },
    )


def run_pipeline():
    with st.spinner(
        "Preparing data, evaluating models and generating predictions..."
    ):
        result = subprocess.run(
            [sys.executable, "-m", "src.pipeline"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )

    if result.returncode == 0:
        st.success("Prediction pipeline completed.")
        st.session_state.pop("pipeline_error", None)
        st.cache_data.clear()
        st.rerun()
    else:
        error_text = (
            (result.stdout or "")
            + "\n"
            + (result.stderr or "")
        )

        st.session_state["pipeline_error"] = error_text[-10000:]

        st.error(
            "The pipeline failed. Open Advanced Tools → "
            "Pipeline diagnostics to inspect the error."
        )


# ---------------------------------------------------------
# SIDEBAR NAVIGATION
# ---------------------------------------------------------

st.sidebar.markdown("# ⚙️ TELIPORT")
st.sidebar.caption("Self-Healing Machines")
st.sidebar.divider()

page_groups = {
    "MAIN WORKSPACE": [
        "Overview",
        "Engine Investigation",
        "Maintenance Work Orders",
    ],
    "KNOWLEDGE": [
        "Maintenance Knowledge Base",
    ],
    "ADVANCED": [
        "Model & Monitoring",
        "AI Evaluation & History",
        "Pipeline Diagnostics",
    ],
}

page_options = [
    page
    for pages in page_groups.values()
    for page in pages
]

selected_page = None

for group, pages in page_groups.items():
    st.sidebar.caption(group)

    for page_name in pages:
        if st.sidebar.button(
            page_name,
            key=f"nav_{page_name}",
            width="stretch",
            type=(
                "primary"
                if st.session_state.get("active_page", "Overview")
                == page_name
                else "secondary"
            ),
        ):
            st.session_state["active_page"] = page_name

    st.sidebar.write("")

if "active_page" not in st.session_state:
    st.session_state["active_page"] = "Overview"

page = st.session_state["active_page"]

if page not in page_options:
    page = "Overview"
    st.session_state["active_page"] = page

st.sidebar.divider()

data_source = metrics.get(
    "data_source",
    "Pipeline not run yet",
)

st.sidebar.caption("DATA SOURCE")
st.sidebar.write(str(data_source).replace("_", " ").title())

st.sidebar.caption("WORKFLOW")
for step in [
    "1. Predict machine condition",
    "2. Prioritize attention",
    "3. Investigate evidence",
    "4. Human review",
    "5. Record outcome",
]:
    st.sidebar.caption(step)

st.sidebar.divider()

if st.sidebar.button(
    "▶ Run Prediction Pipeline",
    type="primary",
    width="stretch",
):
    run_pipeline()

if st.sidebar.button(
    "↻ Refresh Dashboard",
    width="stretch",
):
    refresh_data()


# ---------------------------------------------------------
# SHARED PAGE HEADER
# ---------------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <h1>Self-Healing Machines</h1>
        <p>
            TELIPORT · Predictive maintenance workspace
            · Review recommendations before taking action
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# PAGE 1: OVERVIEW
# =========================================================

if page == "Overview":

    st.subheader("Overview")

    st.markdown(
        '<p class="section-note">'
        'See which engines may need attention and decide what to investigate.'
        '</p>',
        unsafe_allow_html=True,
    )

    if predictions.empty:
        st.info(
            "No prediction results are available yet. "
            "Click **Run Prediction Pipeline** in the sidebar to begin."
        )

        st.markdown("### How TELIPORT works")

        steps = [
            (
                "1. Predict",
                "Machine learning estimates remaining useful life.",
            ),
            (
                "2. Prioritize",
                "Python rules assign an urgency category.",
            ),
            (
                "3. Investigate",
                "Gemini can explain evidence and retrieved references.",
            ),
            (
                "4. Review",
                "A technician reviews and records the decision.",
            ),
        ]

        for title, description in steps:
            st.markdown(
                f"""
                <div class="workflow-step">
                    <strong>{title}</strong><br>
                    {description}
                </div>
                """,
                unsafe_allow_html=True,
            )

    else:
        rul_column = (
            "predicted_rul"
            if "predicted_rul" in predictions.columns
            else None
        )

        anomaly_column = (
            "anomaly"
            if "anomaly" in predictions.columns
            else None
        )

        decision_column = (
            "decision"
            if "decision" in predictions.columns
            else (
                "triage"
                if "triage" in predictions.columns
                else None
            )
        )

        if decision_column:
            needs_review = predictions[decision_column].astype(str).isin(
                [
                    "URGENT_REVIEW",
                    "PLAN_MAINTENANCE",
                    "INSPECT",
                ]
            ).sum()
        else:
            needs_review = 0

        engine_count = (
            predictions["engine_id"].nunique()
            if "engine_id" in predictions.columns
            else len(predictions)
        )

        anomaly_count = (
            predictions[anomaly_column].apply(is_flagged).sum()
            if anomaly_column
            else 0
        )

        average_rul = (
            predictions[rul_column].mean()
            if rul_column
            else None
        )

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Engines Evaluated", int(engine_count))
        col2.metric("Need Review", int(needs_review))
        col3.metric(
            "Average Predicted RUL",
            f"{fmt(average_rul)} cycles"
            if average_rul is not None
            else "—",
        )
        col4.metric("Unusual Patterns", int(anomaly_count))

        st.divider()

        st.subheader("Engines to review first")

        if rul_column:
            review = predictions.copy()

            review = review.sort_values(
                rul_column,
                ascending=True,
            ).head(10)

            show_prediction_table(review)

            st.caption(
                "Lower predicted RUL means fewer estimated operating cycles "
                "remaining. Predictions are estimates, not guarantees."
            )

            left, right = st.columns([1.25, 1])

            with left:
                st.subheader("Predicted remaining life")

                chart_data = predictions.copy()

                if "engine_id" in chart_data.columns:
                    chart_data = chart_data.sort_values(
                        rul_column,
                        ascending=True,
                    ).head(15)

                    chart_data["engine_id"] = (
                        chart_data["engine_id"].astype(str)
                    )

                    chart_data = chart_data.set_index("engine_id")

                st.bar_chart(
                    chart_data[[rul_column]],
                    height=300,
                )

            with right:
                st.subheader("Understand the results")

                st.markdown(
                    """
                    - **Predicted RUL:** estimated remaining operating cycles.
                    - **Unusual pattern:** an outlier flag, not a confirmed failure.
                    - **Need review:** based on the configured triage rules.
                    """
                )

                st.info(
                    "This project currently evaluates historical NASA "
                    "turbofan data. It is not a live industrial monitoring feed."
                )

        else:
            show_prediction_table(predictions)

        st.divider()

        st.subheader("Next step")

        st.write(
            "Choose **Engine Investigation** in the sidebar to inspect "
            "one engine and, when configured, generate a Gemini report."
        )


# =========================================================
# PAGE 2: ENGINE INVESTIGATION
# =========================================================

elif page == "Engine Investigation":

    st.subheader("Engine Investigation")

    st.caption(
        "Inspect one engine's prediction, anomaly status and AI investigation."
    )

    if predictions.empty or "engine_id" not in predictions.columns:
        st.warning(
            "Prediction data is unavailable. Run the pipeline first."
        )

    else:
        engine_ids = sorted(
            predictions["engine_id"].dropna().astype(int).unique().tolist()
        )

        selected_engine = st.selectbox(
            "Select an engine",
            engine_ids,
        )

        selected_rows = predictions[
            predictions["engine_id"].astype(int) == int(selected_engine)
        ]

        if selected_rows.empty:
            st.error("No prediction was found for this engine.")
        else:
            row = selected_rows.iloc[0]

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Predicted RUL",
                f"{fmt(get_rul(row))} cycles",
            )

            anomaly_value = row.get("anomaly", None)

            if anomaly_value is None:
                anomaly_label = "Unavailable"
            else:
                anomaly_label = (
                    "Flagged"
                    if is_flagged(anomaly_value)
                    else "Not flagged"
                )

            col2.metric("Unusual Pattern", anomaly_label)

            decision = get_decision(row)
            col3.metric("Triage Priority", str(decision))

            st.subheader("Engine evidence")
            show_table(pd.DataFrame([row]))

            sensor_plot = REPORTS / "sensor_history.png"

            if sensor_plot.exists():
                with st.expander("View sensor history"):
                    st.image(
                        str(sensor_plot),
                        width="stretch",
                    )
                    st.caption(
                        "Confirm the engine represented by this plot "
                        "before interpreting it as evidence for the selected engine."
                    )

            st.divider()

            st.subheader("AI Maintenance Investigation")

            st.write(
                "Gemini can explain the available evidence and relevant "
                "approved maintenance documents. It cannot change the "
                "ML prediction or approve a maintenance action."
            )

            if st.button(
                "Generate Investigation Report",
                type="primary",
            ):
                try:
                    from src.llm_workflow import analyze_engine

                    with st.spinner(
                        "Retrieving references and generating the report..."
                    ):
                        analysis = analyze_engine(int(selected_engine))

                    st.session_state[
                        f"analysis_{selected_engine}"
                    ] = analysis

                    st.success("Investigation report generated.")

                except Exception as exc:
                    st.error(
                        f"Investigation could not be completed: {exc}"
                    )

            analysis = st.session_state.get(
                f"analysis_{selected_engine}"
            )

            saved_report = (
                REPORTS
                / f"llm_analysis_engine_{int(selected_engine)}.json"
            )

            if analysis is None and saved_report.exists():
                try:
                    analysis = json.loads(
                        saved_report.read_text(encoding="utf-8")
                    )
                except (json.JSONDecodeError, OSError):
                    analysis = None

            if analysis:
                st.subheader("Investigation Result")

                readable_fields = [
                    "summary",
                    "assessment",
                    "possible_causes",
                    "recommended_checks",
                    "limitations",
                    "uncertainty",
                ]

                for field in readable_fields:
                    value = analysis.get(field)

                    if value:
                        st.markdown(
                            f"**{field.replace('_', ' ').title()}**"
                        )
                        st.write(value)

                references = analysis.get(
                    "retrieved_references",
                    [],
                )

                st.markdown("**Supporting references**")

                if references:
                    show_table(pd.DataFrame(references))
                else:
                    st.warning(
                        "No approved maintenance reference was retrieved. "
                        "This report is not backed by a specific manual."
                    )

                with st.expander("Technical report (JSON)"):
                    st.json(analysis)


# =========================================================
# PAGE 3: MAINTENANCE WORK ORDERS
# =========================================================

elif page == "Maintenance Work Orders":

    st.subheader("Maintenance Work Orders")

    st.caption(
        "Manage work orders after reviewing the machine evidence."
    )

    try:
        orders = read_orders()
    except Exception as exc:
        orders = pd.DataFrame()
        st.error(f"Could not load work orders: {exc}")

    if orders.empty:
        st.info(
            "There are no actionable work orders yet. "
            "Run the pipeline to generate triage decisions."
        )

    else:
        show_table(orders)

        if "id" in orders.columns:
            order_ids = orders["id"].dropna().astype(int).tolist()

            selected_order = st.selectbox(
                "Select a work order",
                order_ids,
            )

            selected_rows = orders[
                orders["id"].astype(int) == int(selected_order)
            ]

            selected_record = (
                selected_rows.iloc[0]
                if not selected_rows.empty
                else {}
            )

            current_status = str(
                selected_record.get("status", "OPEN")
            )

            status_options = [
                "OPEN",
                "ASSIGNED",
                "IN_PROGRESS",
                "RESOLVED",
                "CANCELLED",
            ]

            if current_status not in status_options:
                status_options.append(current_status)

            st.subheader("Update selected work order")

            new_status = st.selectbox(
                "Status",
                status_options,
                index=status_options.index(current_status),
            )

            if st.button("Save Status"):
                try:
                    update_status(int(selected_order), new_status)
                    st.success("Status updated.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Could not update status: {exc}")

            technician = st.text_input(
                "Technician or team",
                key=f"assignee_{selected_order}",
            )

            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button("Assign", width="stretch"):
                    if not technician.strip():
                        st.warning("Enter a technician or team name.")
                    else:
                        try:
                            assign_order(
                                int(selected_order),
                                technician.strip(),
                            )
                            st.success("Work order assigned.")
                            st.rerun()
                        except Exception as exc:
                            st.error(str(exc))

            with col2:
                if st.button(
                    "Acknowledge",
                    width="stretch",
                ):
                    try:
                        acknowledge_order(int(selected_order))
                        st.success("Acknowledgement recorded.")
                        st.rerun()
                    except Exception as exc:
                        st.error(str(exc))

            with col3:
                if st.button(
                    "Escalate",
                    width="stretch",
                ):
                    try:
                        escalate_order(int(selected_order))
                        st.warning("Escalation recorded.")
                        st.rerun()
                    except Exception as exc:
                        st.error(str(exc))

            with st.expander("Work-order audit trail"):
                try:
                    audit = read_audit(int(selected_order))
                    show_table(audit)
                except Exception as exc:
                    st.error(f"Could not load audit trail: {exc}")

    with st.expander("All triage decisions"):
        if decisions_path.exists():
            show_table(
                load_csv(
                    str(decisions_path),
                    file_version(decisions_path),
                )
            )
        else:
            st.info("Run the pipeline to generate triage decisions.")


# =========================================================
# PAGE 4: MAINTENANCE KNOWLEDGE BASE
# =========================================================

elif page == "Maintenance Knowledge Base":

    st.subheader("Maintenance Knowledge Base")

    st.caption(
        "Register authorized manuals and have a responsible engineer approve them."
    )

    st.warning(
        "Do not approve a manual until its equipment model, revision, "
        "scope and source have been checked."
    )

    with st.expander("Register a new manual", expanded=True):

        uploaded_file = st.file_uploader(
            "Upload a manual",
            type=["pdf", "docx", "md", "txt"],
        )

        document_title = st.text_input("Document title")
        asset_model = st.text_input("Equipment model")
        revision = st.text_input("Document revision")
        notes = st.text_area("Source and scope notes")

        if st.button("Register for Review", type="primary"):

            required_fields = [
                uploaded_file,
                document_title.strip(),
                asset_model.strip(),
                revision.strip(),
            ]

            if not all(required_fields):
                st.error(
                    "Select a file and complete all required fields."
                )

            else:
                try:
                    from src.maintenance_rag import register_document

                    destination = (
                        ROOT
                        / "knowledge"
                        / "maintenance_docs"
                        / Path(uploaded_file.name).name
                    )

                    destination.parent.mkdir(
                        parents=True,
                        exist_ok=True,
                    )

                    destination.write_bytes(
                        uploaded_file.getvalue()
                    )

                    info = register_document(
                        destination,
                        title=document_title.strip(),
                        asset_model=asset_model.strip(),
                        revision=revision.strip(),
                        notes=notes.strip(),
                    )

                    st.success(
                        "Document registered for review. "
                        f"Document ID: {info['doc_id']}"
                    )

                except Exception as exc:
                    st.error(f"Registration failed: {exc}")

    try:
        from src.maintenance_rag import (
            list_documents,
            approve_document,
            document_audit,
        )

        documents = list_documents()

        st.subheader("Registered Documents")

        if documents:
            show_table(pd.DataFrame(documents))
        else:
            st.info(
                "No documents are registered. "
                "The AI cannot provide manual-backed guidance "
                "until appropriate documents are added and approved."
            )

        pending_documents = [
            document
            for document in documents
            if document.get("status") != "approved"
        ]

        if pending_documents:

            st.subheader("Review a Document")

            document_id = st.selectbox(
                "Pending document",
                [item["doc_id"] for item in pending_documents],
            )

            reviewer = st.text_input(
                "Responsible engineer",
                key="document_reviewer",
            )

            expiry = st.text_input(
                "Expiry date (ISO-8601)",
                value="2027-12-31T23:59:00+00:00",
                help="Replace the example with the actual expiry date.",
            )

            if st.button("Approve Document"):

                try:
                    if document_id is None:
                        raise ValueError(
                            "Select a pending document to approve."
                        )

                    if not reviewer.strip():
                        raise ValueError(
                            "Enter the responsible engineer's name."
                        )

                    approve_document(
                        document_id,
                        reviewer.strip(),
                        expiry.strip(),
                    )

                    st.success("Document approval recorded.")
                    st.rerun()

                except Exception as exc:
                    st.error(f"Approval failed: {exc}")

        with st.expander("Document approval audit trail"):
            events = document_audit()

            if events:
                show_table(pd.DataFrame(events))
            else:
                st.info("No document approval events are recorded.")

    except Exception as exc:
        st.error(f"Knowledge-base functions are unavailable: {exc}")


# =========================================================
# PAGE 5: MODEL AND MONITORING
# =========================================================

elif page == "Model & Monitoring":

    st.subheader("Model Performance & Monitoring")

    st.caption(
        "Technical information for evaluating the predictive model."
    )

    if metrics:

        test_metrics = metrics.get("test", {})
        validation_metrics = metrics.get(
            "random_forest_validation",
            {},
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Test Engines",
            metrics.get("test_engine_count", "—"),
        )

        col2.metric(
            "Test MAE",
            fmt(test_metrics.get("mae")),
        )

        col3.metric(
            "Validation MAE",
            fmt(validation_metrics.get("mae")),
        )

        st.write(
            "**Dataset source:**",
            metrics.get("data_source", "Unknown"),
        )

        st.caption(
            metrics.get(
                "note",
                "Metrics describe the evaluated dataset and "
                "do not guarantee real-machine performance.",
            )
        )

        with st.expander("Baseline vs. model comparison"):
            comparison = {
                "Baseline": metrics.get(
                    "baseline_validation",
                    {},
                ),
                "Random Forest": validation_metrics,
            }

            show_table(pd.DataFrame(comparison).T)

        with st.expander("Data quality report"):
            st.json(metrics.get("data_quality", {}))

        with st.expander("Full metrics JSON"):
            st.json(metrics)

    else:
        st.info(
            "No metrics are available. Run the prediction pipeline first."
        )

    candidate_path = REPORTS / "candidate_model.json"

    if candidate_path.exists():

        try:
            candidate = json.loads(
                candidate_path.read_text(encoding="utf-8")
            )
        except (json.JSONDecodeError, OSError):
            candidate = {}

        st.divider()
        st.subheader("Candidate Model Review")

        st.write(
            "A candidate model is a newly trained model awaiting review. "
            "It is not automatically the approved production model."
        )

        st.write(
            "**Candidate status:**",
            candidate.get("status", "Not specified"),
        )

        with st.expander("Candidate technical details"):
            st.json(candidate)

        approver = st.text_input(
            "Approver name",
            key="candidate_approver",
        )

        rationale = st.text_area(
            "Reason for approval or rejection",
            key="candidate_rationale",
        )

        col1, col2 = st.columns(2)

        with col1:
            if st.button(
                "Approve Candidate",
                type="primary",
                width="stretch",
            ):

                if not approver.strip():
                    st.error("Enter the approver's name.")

                else:
                    try:
                        approve_candidate(
                            approver.strip(),
                            True,
                            rationale.strip(),
                        )

                        st.success("Candidate approval recorded.")
                        st.rerun()

                    except Exception as exc:
                        st.error(f"Approval failed: {exc}")

        with col2:
            if st.button(
                "Reject Candidate",
                width="stretch",
            ):

                if not approver.strip():
                    st.error("Enter the approver's name.")

                else:
                    try:
                        approve_candidate(
                            approver.strip(),
                            False,
                            rationale.strip(),
                        )

                        st.warning(
                            "Candidate rejection recorded. "
                            "The existing approved model should remain unchanged."
                        )

                        st.rerun()

                    except Exception as exc:
                        st.error(f"Rejection failed: {exc}")

    monitoring_path = REPORTS / "monitoring.json"

    if monitoring_path.exists():

        st.divider()
        st.subheader("Data Drift Monitoring")

        try:
            monitoring = json.loads(
                monitoring_path.read_text(encoding="utf-8")
            )

            with st.expander("Monitoring details"):
                st.json(monitoring)

        except (json.JSONDecodeError, OSError):
            st.warning("The monitoring report could not be read.")

    st.divider()

    st.subheader("MLflow")

    st.write(
        "MLflow can record training experiments, parameters, "
        "metrics and model artifacts when installed and configured."
    )

    st.code(
        "mlflow ui --backend-store-uri ./mlruns",
        language="bash",
    )


# =========================================================
# PAGE 6: AI EVALUATION AND HISTORY
# =========================================================

elif page == "AI Evaluation & History":

    st.subheader("AI Evaluation & Investigation History")

    st.caption(
        "Review AI quality, saved investigations and technician feedback."
    )

    with st.expander("Run RAG evaluation"):

        st.write(
            "The evaluation script tests document retrieval. "
            "Meaningful accuracy scores require human-reviewed expected sources."
        )

        if st.button("Run RAG Evaluation"):

            evaluation_script = ROOT / "scripts" / "evaluate_rag.py"

            if not evaluation_script.exists():
                st.error("The RAG evaluation script was not found.")

            else:
                with st.spinner("Running RAG evaluation..."):
                    result = subprocess.run(
                        [
                            sys.executable,
                            str(evaluation_script),
                        ],
                        cwd=ROOT,
                        capture_output=True,
                        text=True,
                    )

                if result.returncode == 0:
                    st.success("Evaluation completed.")
                else:
                    st.error("Evaluation failed.")

                if result.stdout:
                    st.code(result.stdout)

                if result.stderr:
                    with st.expander("Evaluation diagnostics"):
                        st.code(result.stderr)

        st.markdown("**LangSmith evaluation**")

        st.code(
            "python scripts/push_langsmith_eval.py",
            language="bash",
        )

        st.caption(
            "LangSmith requires credentials. Check data privacy before "
            "sending prompts or evaluation examples to an external service."
        )

    try:
        from src.investigation_store import (
            list_investigations,
            submit_feedback,
        )

        history = list_investigations()

        st.subheader("Saved Investigations")

        if not history:
            st.info(
                "No saved investigations yet. "
                "Open Engine Investigation to generate a report."
            )

        else:
            history_frame = pd.DataFrame(history)
            show_table(history_frame)

            investigation_ids = [
                int(item["id"])
                for item in history
            ]

            with st.form("investigation_feedback_form"):

                st.subheader("Record Technician Feedback")

                investigation_id = st.selectbox(
                    "Investigation",
                    investigation_ids,
                )

                reviewer_name = st.text_input(
                    "Technician or reviewer",
                )

                helpful = st.checkbox(
                    "Was the investigation useful?",
                )

                correctness = st.selectbox(
                    "Technical assessment",
                    [
                        "NOT_ASSESSED",
                        "CORRECT",
                        "PARTIALLY_CORRECT",
                        "INCORRECT",
                        "UNVERIFIABLE",
                    ],
                )

                review_status = st.selectbox(
                    "Review outcome",
                    [
                        "REVIEWED",
                        "NEEDS_CORRECTION",
                        "REJECTED",
                    ],
                )

                feedback_text = st.text_area(
                    "Feedback or observed outcome",
                )

                submitted = st.form_submit_button(
                    "Save Feedback",
                    type="primary",
                )

                if submitted:

                    if not reviewer_name.strip():
                        st.error("Enter the reviewer name.")

                    else:
                        try:
                            submit_feedback(
                                investigation_id,
                                reviewer_name.strip(),
                                helpful,
                                correctness,
                                feedback_text,
                                review_status,
                            )

                            st.success("Feedback saved.")
                            st.rerun()

                        except Exception as exc:
                            st.error(
                                f"Feedback could not be saved: {exc}"
                            )

    except Exception as exc:
        st.error(
            f"Investigation history is unavailable: {exc}"
        )


# =========================================================
# PAGE 7: PIPELINE DIAGNOSTICS
# =========================================================

elif page == "Pipeline Diagnostics":

    st.subheader("Pipeline Diagnostics")

    st.caption(
        "Use this page only when the prediction pipeline fails or "
        "technical debugging is needed."
    )

    if "pipeline_error" in st.session_state:

        st.error("The most recent pipeline run failed.")

        st.code(
            st.session_state["pipeline_error"],
            language="text",
        )

        if st.button("Clear Error Message"):
            st.session_state.pop("pipeline_error", None)
            st.rerun()

    else:
        st.success(
            "No pipeline error is recorded in this dashboard session."
        )

    st.subheader("Project files")

    important_files = {
        "Metrics": metrics_path,
        "Predictions": predictions_path,
        "Triage decisions": decisions_path,
        "Candidate model": REPORTS / "candidate_model.json",
        "Monitoring": REPORTS / "monitoring.json",
    }

    status_rows = []

    for label, path in important_files.items():
        status_rows.append(
            {
                "Artifact": label,
                "Status": "Available" if path.exists() else "Not generated",
                "Path": str(path.relative_to(ROOT))
                if path.is_relative_to(ROOT)
                else path.name,
            }
        )

    show_table(pd.DataFrame(status_rows))

    st.subheader("Run the pipeline manually")

    st.code(
        "python -m src.pipeline",
        language="bash",
    )

    st.subheader("Run tests")

    st.code(
        "python -m pytest -q",
        language="bash",
    )


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.divider()

st.caption(
    "TELIPORT prototype · RUL predictions are estimates. "
    "Anomaly flags are not confirmed failures. "
    "Maintenance actions require qualified human review."
)
