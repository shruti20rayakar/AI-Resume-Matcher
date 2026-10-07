from io import BytesIO
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import streamlit as st
import tempfile

from hybrid_matcher import generate_hybrid_ranking
from explanation_engine import generate_explanation


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Resume Job Matcher",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# CUSTOM STYLING
# ============================================================
st.markdown(
    """
    <style>

    /* Main page */
    .main {
        padding-top: 2rem;
    }

    /* Title */
    .main-title {
        font-size: 46px;
        font-weight: 800;
        text-align: center;
        margin-bottom: 8px;
        letter-spacing: -1px;
    }

    .subtitle {
        text-align: center;
        font-size: 19px;
        opacity: 0.75;
        margin-bottom: 40px;
    }

    /* Section headers */
    .section-header {
        font-size: 30px;
        font-weight: 750;
        margin-top: 25px;
        margin-bottom: 15px;
    }

    /* Job cards */
    .job-card {
        padding: 24px;
        border: 1px solid rgba(128,128,128,0.25);
        border-radius: 16px;
        margin-bottom: 22px;
        background: rgba(128,128,128,0.05);
    }

    .job-title {
        font-size: 24px;
        font-weight: 700;
    }

    .company {
        font-size: 16px;
        opacity: 0.75;
        margin-top: 4px;
    }

    /* Match score */
    .score {
        font-size: 34px;
        font-weight: 800;
        text-align: right;
    }

    .verdict {
        text-align: right;
        font-size: 15px;
        font-weight: 600;
        opacity: 0.8;
    }

    /* Score labels */
    .metric-label {
        font-size: 13px;
        opacity: 0.7;
    }

    /* Upload section */
    .upload-box {
        padding: 25px;
        border-radius: 16px;
        border: 1px dashed rgba(128,128,128,0.5);
        margin-bottom: 25px;
    }

    /* Footer */
    .footer {
        text-align: center;
        opacity: 0.6;
        margin-top: 50px;
        padding: 20px;
        font-size: 13px;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">'
    '🤖 AI Resume Job Matcher'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Hybrid AI system for intelligent resume-job matching'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("About")

    st.write(
        """
        This application analyzes a resume and ranks
        job opportunities using multiple signals.
        """
    )

    st.write("### Matching Components")

    st.write(
        """
        • TF-IDF similarity  
        • Skill matching  
        • Transformer semantic similarity  
        • Experience relevance  
        • Education relevance
        """
    )

    st.write("### Output")

    st.write(
        """
        • Overall match score  
        • Ranked jobs  
        • Matched skills  
        • Missing skills  
        • Score explanation
        """
    )


# ============================================================
# RESUME UPLOAD
# ============================================================

st.header("📄 Upload Your Resume")

uploaded_file = st.file_uploader(
    "Upload a PDF or DOCX resume",
    type=["pdf", "docx"]
)


# ============================================================
# ANALYZE BUTTON
# ============================================================

if uploaded_file is not None:

    st.success(
        f"Resume uploaded: {uploaded_file.name}"
    )

    analyze_button = st.button(
        "🚀 Analyze Resume",
        type="primary"
    )

    if analyze_button:

        # ----------------------------------------------------
        # CREATE TEMPORARY FILE
        # ----------------------------------------------------

        file_extension = (
            Path(
                uploaded_file.name
            ).suffix
        )

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=file_extension
        ) as temp_file:

            temp_file.write(
                uploaded_file.getbuffer()
            )

            temp_resume_path = (
                temp_file.name
            )

        # ----------------------------------------------------
        # RUN MODEL
        # ----------------------------------------------------

        with st.spinner(
            "Analyzing resume and matching jobs..."
        ):

            try:

                results, resume_data = (
                    generate_hybrid_ranking(
                        temp_resume_path
                    )
                )

            except Exception as error:

                st.error(
                    "An error occurred while "
                    "analyzing the resume."
                )

                st.exception(error)

                st.stop()

        # ----------------------------------------------------
        # RESUME SUMMARY
        # ----------------------------------------------------

        st.header("👤 Resume Analysis")

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("Candidate")

            st.write(
                resume_data.get(
                    "name",
                    "Not detected"
                )
            )

        with col2:

            st.subheader("Skills Detected")

            skills = resume_data.get(
                "skills",
                []
            )

            if skills:

                st.write(
                    ", ".join(skills)
                )

            else:

                st.write(
                    "No skills detected."
                )

        st.divider()

        # ----------------------------------------------------
        # TOP JOBS
        # ----------------------------------------------------

        st.header(
            "🏆 Top Job Matches"
        )

                # ----------------------------------------------------
        # TOP JOBS
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-header">'
            '🏆 Top Job Matches'
            '</div>',
            unsafe_allow_html=True
        )

        top_jobs = results.head(10)

        for rank, (_, row) in enumerate(
            top_jobs.iterrows(),
            start=1
        ):

            explanation = generate_explanation(row)

            score = float(row["final_score"])

            verdict = explanation["verdict"]

            # ------------------------------------------------
            # RANK BADGE
            # ------------------------------------------------

            if rank == 1:
                badge = "🥇"
            elif rank == 2:
                badge = "🥈"
            elif rank == 3:
                badge = "🥉"
            else:
                badge = f"#{rank}"

            # ------------------------------------------------
            # JOB CARD
            # ------------------------------------------------

            st.markdown(
                '<div class="job-card">',
                unsafe_allow_html=True
            )

            title_col, score_col = st.columns(
                [3.5, 1]
            )

            with title_col:

                st.markdown(
                    f'<div class="job-title">'
                    f'{badge} {row["title"]}'
                    f'</div>',
                    unsafe_allow_html=True
                )

                st.markdown(
                    f'<div class="company">'
                    f'🏢 {row["company"]}'
                    f'</div>',
                    unsafe_allow_html=True
                )

            with score_col:

                st.markdown(
                    f'<div class="score">'
                    f'{score:.1f}%'
                    f'</div>',
                    unsafe_allow_html=True
                )

                st.markdown(
                    f'<div class="verdict">'
                    f'{verdict}'
                    f'</div>',
                    unsafe_allow_html=True
                )

            # ------------------------------------------------
            # MATCH PROGRESS
            # ------------------------------------------------

            st.progress(
                min(
                    max(score / 100, 0.0),
                    1.0
                )
            )

            # ------------------------------------------------
            # SCORE BREAKDOWN
            # ------------------------------------------------

            st.markdown(
                "#### 📊 Score Breakdown"
            )

            c1, c2, c3, c4, c5 = st.columns(5)

            with c1:

                st.metric(
                    "TF-IDF",
                    f"{row['tfidf_score']:.1f}%"
                )

            with c2:

                st.metric(
                    "Skills",
                    f"{row['skill_score']:.1f}%"
                )

            with c3:

                st.metric(
                    "Semantic",
                    f"{row['semantic_score']:.1f}%"
                )

            with c4:

                st.metric(
                    "Experience",
                    f"{row['experience_score']:.1f}%"
                )

            with c5:

                st.metric(
                    "Education",
                    f"{row['education_score']:.1f}%"
                )

            # ------------------------------------------------
            # EXPANDABLE DETAILS
            # ------------------------------------------------

            with st.expander(
                "🔍 View Matching Details"
            ):

                skill_col1, skill_col2 = st.columns(2)

                # --------------------------------------------
                # MATCHED SKILLS
                # --------------------------------------------

                with skill_col1:

                    st.markdown(
                        "##### ✅ Matched Required Skills"
                    )

                    matched = explanation[
                        "matched_required"
                    ]

                    if matched:

                        for skill in matched:

                            st.write(
                                f"✓ {skill}"
                            )

                    else:

                        st.write(
                            "No required skills matched."
                        )

                # --------------------------------------------
                # MISSING SKILLS
                # --------------------------------------------

                with skill_col2:

                    st.markdown(
                        "##### ⚠️ Missing Required Skills"
                    )

                    missing = explanation[
                        "missing_required"
                    ]

                    if missing:

                        for skill in missing:

                            st.write(
                                f"⚠ {skill}"
                            )

                    else:

                        st.write(
                            "None — all required skills matched."
                        )

                st.divider()

                # --------------------------------------------
                # PREFERRED SKILLS
                # --------------------------------------------

                st.markdown(
                    "##### ⭐ Preferred Skills Matched"
                )

                preferred = explanation[
                    "matched_preferred"
                ]

                if preferred:

                    st.write(
                        ", ".join(preferred)
                    )

                else:

                    st.write(
                        "No preferred skills matched."
                    )

                # --------------------------------------------
                # WHY THIS MATCH
                # --------------------------------------------

                st.markdown(
                    "##### 💡 Why This Match?"
                )

                for reason in explanation[
                    "score_explanation"
                ]:

                    st.write(
                        f"• {reason}"
                    )

            st.markdown(
                "</div>",
                unsafe_allow_html=True
            )
        # ----------------------------------------------------
        # ----------------------------------------------------
        # DOWNLOAD MATCHING REPORT
        # ----------------------------------------------------

        st.divider()

        st.markdown(
            '<div class="section-header">'
            '📥 Download Matching Report'
            '</div>',
            unsafe_allow_html=True
        )

        report_lines = [
            "AI RESUME-JOB MATCHING REPORT",
            "=" * 60,
            "",
            f"Candidate: {resume_data.get('name', 'Unknown')}",
            "",
            "Detected Skills:",
            ", ".join(resume_data.get("skills", [])),
            "",
            "TOP JOB MATCHES",
            "-" * 60,
        ]

        for rank, (_, row) in enumerate(
            top_jobs.iterrows(),
            start=1
        ):
            explanation = generate_explanation(row)

            report_lines.extend([
                "",
                f"{rank}. {row['title']}",
                f"Company: {row['company']}",
                f"Verdict: {explanation['verdict']}",
                f"Final Match: {float(row['final_score']):.2f}%",
                f"TF-IDF: {float(row['tfidf_score']):.2f}%",
                f"Skills: {float(row['skill_score']):.2f}%",
                f"Semantic: {float(row['semantic_score']):.2f}%",
                f"Experience: {float(row['experience_score']):.2f}%",
                f"Education: {float(row['education_score']):.2f}%",
                "",
                "Matched Required Skills:",
                ", ".join(explanation["matched_required"])
                if explanation["matched_required"] else "None",
                "",
                "Missing Required Skills:",
                ", ".join(explanation["missing_required"])
                if explanation["missing_required"] else "None",
                "",
                "Why This Match:",
            ])

            for reason in explanation["score_explanation"]:
                report_lines.append(f"- {reason}")

            report_lines.append("-" * 60)

        report_text = "\n".join(report_lines)

        st.download_button(
            label="📄 Download Matching Report",
            data=report_text,
            file_name="resume_matching_report.txt",
            mime="text/plain",
            key="download_matching_report"
        )

        # CLEANUP
        # ----------------------------------------------------

        try:

            Path(
                temp_resume_path
            ).unlink()

        except Exception:
            pass

else:

    st.info(
        "Upload your resume above to begin."
    )