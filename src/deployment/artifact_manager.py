"""Hugging Face Space artifact download manager.

Manages check and automated download of FAISS index, lookup metadata, and
candidate profile database from Hugging Face Dataset registries.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional

from src.config import (
    HF_ARTIFACT_DIR,
    HF_CANDIDATE_DATASET_REPO,
    HF_CANDIDATE_DATASET_REVISION,
    HF_CANDIDATE_FILENAME,
    HF_DATASET_REPO,
    HF_DATASET_REVISION,
    HF_DOWNLOAD_TIMEOUT,
    HF_FORCE_DOWNLOAD,
    PROJECT_ROOT,
)

logger = logging.getLogger(__name__)

# List of required FAISS filenames
REQUIRED_FILENAMES = [
    "faiss.index",
    "candidate_lookup.pkl",
    "embedding_metadata.pkl",
]


class ArtifactManager:
    """Manager class for checking and downloading deployment artifacts."""

    @staticmethod
    def ensure_artifacts(*, streamlit_ui: bool = True) -> bool:
        """Verify presence of FAISS artifacts and candidates.jsonl, downloading missing ones.

        Args:
            streamlit_ui: If True, uses Streamlit status components to display progress.

        Returns:
            True if all artifacts are verified or successfully downloaded.
        """
        logger.info("Checking deployment artifacts...")

        # Determine if we are running inside Streamlit
        is_streamlit = False
        if streamlit_ui:
            try:
                import streamlit as st
                is_streamlit = st.runtime.exists()
            except ImportError:
                pass

        # Identify missing artifacts
        missing_tasks = []

        # Check FAISS artifacts
        for filename in REQUIRED_FILENAMES:
            path = HF_ARTIFACT_DIR / filename
            if HF_FORCE_DOWNLOAD or not path.exists() or path.stat().st_size == 0:
                missing_tasks.append({
                    "repo_id": HF_DATASET_REPO,
                    "filename": filename,
                    "revision": HF_DATASET_REVISION,
                    "dest_dir": HF_ARTIFACT_DIR,
                })

        # Check candidates.jsonl (check root and nested locations to avoid redundant downloads)
        candidate_root_path = PROJECT_ROOT / HF_CANDIDATE_FILENAME
        nested_candidates_path = (
            PROJECT_ROOT
            / "[PUB] India_runs_data_and_ai_challenge"
            / "[PUB] India_runs_data_and_ai_challenge"
            / "India_runs_data_and_ai_challenge"
            / HF_CANDIDATE_FILENAME
        )
        if HF_FORCE_DOWNLOAD or (not candidate_root_path.exists() and not nested_candidates_path.exists()):
            missing_tasks.append({
                "repo_id": HF_CANDIDATE_DATASET_REPO,
                "filename": HF_CANDIDATE_FILENAME,
                "revision": HF_CANDIDATE_DATASET_REVISION,
                "dest_dir": PROJECT_ROOT,
            })

        if not missing_tasks:
            logger.info("Artifacts already present.")
            return True

        # Perform downloading
        missing_names = [task["filename"] for task in missing_tasks]
        logger.info("Missing artifacts identified: %s", missing_names)

        status_container = None
        progress_bar = None

        if is_streamlit:
            import streamlit as st
            status_container = st.empty()
            progress_bar = st.progress(0)
            status_container.markdown(
                "### 🔄 Preparing search index...\n"
                "Downloading deployment artifacts. This runs once on first startup."
            )

        try:
            from huggingface_hub import hf_hub_download
            from huggingface_hub.utils import HfHubHTTPError

            total_files = len(missing_tasks)
            for idx, task in enumerate(missing_tasks):
                filename = task["filename"]
                dest_dir = task["dest_dir"]
                repo_id = task["repo_id"]
                revision = task["revision"]

                logger.info("Downloading %s...", filename)

                if is_streamlit:
                    status_container.markdown(
                        f"### 🔄 Preparing search index...\n"
                        f"Downloading **{filename}** ({idx + 1}/{total_files})..."
                    )
                    progress_bar.progress(int((idx / total_files) * 100))

                # Ensure destination folder exists
                dest_dir.mkdir(parents=True, exist_ok=True)

                # Download file
                hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    repo_type="dataset",
                    revision=revision,
                    local_dir=dest_dir,
                    etag_timeout=HF_DOWNLOAD_TIMEOUT,
                )

            logger.info("Deployment artifacts ready.")

            if is_streamlit:
                progress_bar.progress(100)
                status_container.success("✅ Search index ready.")
                import time
                time.sleep(1.0)
                progress_bar.empty()
                status_container.empty()

            return True

        except Exception as exc:
            err_msg = f"Failed to download deployment artifacts: {exc}"
            logger.error(err_msg)

            if is_streamlit:
                import streamlit as st
                if progress_bar:
                    progress_bar.empty()
                if status_container:
                    status_container.error(
                        "⚠️ **Deployment Artifacts Error**\n\n"
                        "The application could not download deployment artifacts from Hugging Face. "
                        "Please verify your internet connection or space setup.\n\n"
                        f"*Details: {exc}*"
                    )
                st.stop()
            else:
                print(f"ERROR: {err_msg}", file=sys.stderr)
                sys.exit(1)
            return False
