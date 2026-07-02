"""Hugging Face Space artifact download manager.

Manages check and automated download of FAISS index and lookup metadata
artifacts from the Hugging Face Dataset hub.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional

from src.config import (
    HF_ARTIFACT_DIR,
    HF_DATASET_REPO,
    HF_DATASET_REVISION,
    HF_DOWNLOAD_TIMEOUT,
    HF_FORCE_DOWNLOAD,
)

logger = logging.getLogger(__name__)

# List of required filenames to load the index
REQUIRED_FILENAMES = [
    "faiss.index",
    "candidate_lookup.pkl",
    "embedding_metadata.pkl",
]


class ArtifactManager:
    """Manager class for checking and downloading deployment artifacts."""

    @staticmethod
    def ensure_artifacts(*, streamlit_ui: bool = True) -> bool:
        """Verify presence of FAISS artifacts, downloading missing ones if necessary.

        Args:
            streamlit_ui: If True, uses Streamlit status components to display progress.

        Returns:
            True if artifacts are successfully verified or downloaded.
        """
        logger.info("Checking deployment artifacts...")

        # Determine if we are running inside Streamlit
        is_streamlit = False
        if streamlit_ui:
            try:
                import streamlit as st
                # Check if Streamlit is running/initialized
                is_streamlit = st.runtime.exists()
            except ImportError:
                pass

        # Identify missing files
        missing_files = []
        for filename in REQUIRED_FILENAMES:
            path = HF_ARTIFACT_DIR / filename
            if HF_FORCE_DOWNLOAD or not path.exists() or path.stat().st_size == 0:
                missing_files.append(filename)

        if not missing_files:
            logger.info("Artifacts already present.")
            return True

        # Perform downloading
        logger.info("Missing artifacts identified: %s", missing_files)

        status_container = None
        progress_bar = None

        if is_streamlit:
            import streamlit as st
            status_container = st.empty()
            progress_bar = st.progress(0)
            status_container.markdown(
                "### 🔄 Preparing search index...\n"
                "Downloading deployment artifacts from Hugging Face Dataset. This runs once on first startup."
            )

        try:
            from huggingface_hub import hf_hub_download
            from huggingface_hub.utils import HfHubHTTPError

            # Create destination folder if not exists
            HF_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

            total_files = len(missing_files)
            for idx, filename in enumerate(missing_files):
                dest_path = HF_ARTIFACT_DIR / filename
                logger.info("Downloading %s...", filename)

                if is_streamlit:
                    status_container.markdown(
                        f"### 🔄 Preparing search index...\n"
                        f"Downloading file **{filename}** ({idx + 1}/{total_files})..."
                    )
                    progress_bar.progress(int((idx / total_files) * 100))

                # Download file
                hf_hub_download(
                    repo_id=HF_DATASET_REPO,
                    filename=filename,
                    repo_type="dataset",
                    revision=HF_DATASET_REVISION,
                    local_dir=HF_ARTIFACT_DIR,
                    etag_timeout=HF_DOWNLOAD_TIMEOUT,
                )

            logger.info("Deployment artifacts ready.")

            if is_streamlit:
                progress_bar.progress(100)
                status_container.success("✅ Search index ready.")
                # Brief sleep to let the user see the success message
                import time
                time.sleep(1.0)
                progress_bar.empty()
                status_container.empty()

            return True

        except Exception as exc:
            err_msg = f"Failed to download deployment artifacts from {HF_DATASET_REPO}: {exc}"
            logger.error(err_msg)

            if is_streamlit:
                import streamlit as st
                if progress_bar:
                    progress_bar.empty()
                if status_container:
                    status_container.error(
                        "⚠️ **Deployment Artifacts Error**\n\n"
                        "The application could not download search index files from Hugging Face. "
                        "Please verify your internet connection or space setup.\n\n"
                        f"*Details: {exc}*"
                    )
                st.stop()
            else:
                print(f"ERROR: {err_msg}", file=sys.stderr)
                sys.exit(1)
            return False
