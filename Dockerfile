# Use python 3.11 lightweight slim image
FROM python:3.11-slim

# Install system dependencies required by FAISS and other libraries (e.g. libgomp for openmp support)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set up a non-root user with UID 1000 (Hugging Face Spaces default)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set the working directory in the user's home
WORKDIR $HOME/app

# Copy requirements and install dependencies
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy all application files (excluding those in .dockerignore)
COPY --chown=user . .

# Expose the port Streamlit will listen on (Hugging Face default is 7860)
EXPOSE 7860

# Streamlit environment configurations to disable telemetry, set port, and disable file watcher
ENV STREAMLIT_SERVER_PORT=7860 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ENABLE_CORS=false \
    STREAMLIT_SERVER_FILE_WATCHER_TYPE=none

# Run the Streamlit app
CMD ["streamlit", "run", "app.py"]
