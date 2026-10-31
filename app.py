import streamlit as st
from pathlib import Path
import uuid

from main import run_pipeline


# -----------------------------------
# PAGE CONFIGURATION
# -----------------------------------

st.set_page_config(
    page_title="Evidence-Grounded AI",
    page_icon="🔎",
    layout="wide"
)


# -----------------------------------
# CUSTOM HEADER
# -----------------------------------

st.title("🔎 Evidence-Grounded AI")
st.subheader(
    "Question Answering and Hallucination Detection System"
)

st.markdown(
    """
    Ask a question and receive an answer based on retrieved evidence.
    The system also checks individual answer claims against the evidence.
    """
)

st.divider()


# -----------------------------------
# SIDEBAR
# -----------------------------------

st.sidebar.title("⚙️ Search Settings")

search_mode_label = st.sidebar.radio(
    "Select Search Mode",
    [
        "General Search (Web + Online PDFs)",
        "Uploaded PDF Only",
        "Combined Search (PDF + Web + Online PDFs)"
    ]
)

mode_mapping = {
    "General Search (Web + Online PDFs)": "general",
    "Uploaded PDF Only": "uploaded",
    "Combined Search (PDF + Web + Online PDFs)": "combined"
}

search_mode = mode_mapping[search_mode_label]

st.sidebar.info(
    """
    **General Search:** Searches webpages and online PDFs.

    **Uploaded PDF Only:** Searches your uploaded document.

    **Combined Search:** Uses uploaded PDF, webpages and online PDFs.
    """
)


# -----------------------------------
# QUESTION INPUT
# -----------------------------------

st.header("💬 Ask Your Question")

question = st.text_area(
    "Enter your question:",
    placeholder="Example: How are mountains formed?",
    height=110
)


# -----------------------------------
# PDF UPLOAD
# -----------------------------------

uploaded_file = None
uploaded_pdf_path = None

if search_mode in ("uploaded", "combined"):

    st.subheader("📄 Upload Your PDF")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"]
    )

    if uploaded_file is not None:

        st.success(
            f"Uploaded: {uploaded_file.name}"
        )


# -----------------------------------
# RUN BUTTON
# -----------------------------------

run_button = st.button(
    "🚀 Generate Answer & Verify Claims",
    type="primary",
    use_container_width=True
)


# -----------------------------------
# PIPELINE EXECUTION
# -----------------------------------

if run_button:

    if not question.strip():

        st.error("Please enter a question.")

    elif search_mode in ("uploaded", "combined") and uploaded_file is None:

        st.error("Please upload a PDF file first.")

    else:

        try:

            # Save uploaded PDF safely
            if uploaded_file is not None:

                upload_folder = Path("data/uploads")
                upload_folder.mkdir(
                    parents=True,
                    exist_ok=True
                )

                unique_name = (
                    uuid.uuid4().hex + ".pdf"
                )

                uploaded_pdf_path = (
                    upload_folder / unique_name
                )

                uploaded_pdf_path.write_bytes(
                    uploaded_file.getvalue()
                )

            # Run backend
            with st.spinner(
                "Searching evidence, generating answer and verifying claims. Please wait..."
            ):

                result = run_pipeline(
                    question=question.strip(),
                    uploaded_pdf_path=uploaded_pdf_path,
                    search_mode=search_mode
                )

            st.session_state["qa_result"] = result

            st.success("Pipeline completed!")

        except Exception as error:

            st.error(
                f"An error occurred: {error}"
            )


# -----------------------------------
# DISPLAY RESULTS
# -----------------------------------

if "qa_result" in st.session_state:

    result = st.session_state["qa_result"]

    st.divider()

    # ANSWER
    st.header("📝 Generated Answer")

    st.markdown(result.get("answer", "No answer available."))

    # EVIDENCE SUMMARY
    st.divider()

    st.header("📊 Evidence Summary")

    summary = result.get("evidence_summary", {})

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric(
        "Uploaded PDF",
        summary.get("uploaded_pdf", 0)
    )

    col2.metric(
        "Webpages",
        summary.get("webpages", 0)
    )

    col3.metric(
        "Web Snippets",
        summary.get("web_snippets", 0)
    )

    col4.metric(
        "Online PDFs",
        summary.get("online_pdfs", 0)
    )

    col5.metric(
        "Total Evidence",
        summary.get("total", 0)
    )

    # SOURCES
    st.divider()

    st.header("📚 Relevant Sources")

    sources = result.get("sources", [])

    if sources:

        for number, source in enumerate(sources, start=1):

            title = source.get("title", "Unknown Source")
            page = source.get("page", "")
            url = source.get("url", "")
            source_type = source.get("source_type", "Unknown")
            relevance = source.get("relevance_score", 0)

            with st.expander(
                f"Source {number}: {title}"
            ):

                st.write("**Title:**", title)
                st.write("**Page:**", page)
                st.write("**Type:**", source_type)
                st.write(
                    "**Relevance Score:**",
                    round(float(relevance or 0), 4)
                )

                if url:

                    st.markdown(
                        f"[Open Original Source]({url})"
                    )

                if source.get("text"):
                    st.write("**Retrieved Evidence:**")
                    st.write(source["text"])

    else:

        st.info("No relevant sources were returned.")

    # CLAIM VERIFICATION
    st.divider()

    st.header("🛡️ Hallucination Detection Report")

    status_counts = result.get("status_counts", {})

    total_claims = sum(status_counts.values())

    st.write(
        f"**Total Claims Checked:** {total_claims}"
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Supported",
        status_counts.get("Supported", 0)
    )

    c2.metric(
        "Partially Supported",
        status_counts.get("Partially Supported", 0)
    )

    c3.metric(
        "Unsupported",
        status_counts.get("Unsupported", 0)
    )

    c4.metric(
        "Contradicted",
        status_counts.get("Contradicted", 0)
    )

    st.subheader("Individual Claim Verification")

    verification_results = result.get(
        "verification_results", []
    )

    if verification_results:

        for number, item in enumerate(
            verification_results,
            start=1
        ):

            status = item.get("status", "Unverified")

            with st.expander(
                f"Claim {number} — {status}"
            ):

                st.write(
                    "**Claim:**",
                    item.get("claim", "")
                )

                st.write(
                    "**Verification Status:**",
                    status
                )

                confidence = item.get("confidence", 0)

                st.write(
                    "**Confidence:**",
                    confidence
                )

                st.write(
                    "**Reason:**",
                    item.get("reason", "")
                )

                st.write(
                    "**Supporting Evidence:**",
                    item.get("evidence", "Not available")
                )

                st.write(
                    "**Evidence Page:**",
                    item.get("page", "")
                )

                if item.get("title"):
                    st.write(
                        "**Evidence Source:**",
                        item["title"]
                    )

                if item.get("url"):
                    st.markdown(
                        f"[Open Evidence Source]({item['url']})"
                    )

    else:

        st.info("No claims were extracted for verification.")

    # WARNINGS
    errors = result.get("errors", [])

    if errors:

        st.divider()
        st.warning("Some sources could not be processed.")

        for error in errors:
            st.write(f"- {error}")

    # DISCLAIMER
    st.divider()

    st.caption(
        """
        Disclaimer: Verification labels are model-generated assessments
        based on retrieved evidence. They are not a guarantee of factual
        correctness. Please consult the original sources for confirmation.
        """
    )