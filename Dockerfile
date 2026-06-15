# Use Node.js LTS on Debian Trixie (slim variant for smaller image size)
FROM node:lts-trixie-slim

# Install system dependencies needed for development and VS Code devcontainer
# - git: version control (essential for most development workflows)
# - curl: downloading files and API calls
# - ca-certificates: SSL certificate validation
# - gnupg: GPG key management
# - libopenblas-dev/gfortran: gives PyTensor a real BLAS/LAPACK to link against,
#   without which linear algebra ops (and thus PyMC sampling) fall back to a
#   severely degraded pure-Python implementation
RUN apt-get update && apt-get install -y \
    git \
    curl \
    ca-certificates \
    gnupg \
    python3 \
    python3-pip \
    python3.13-venv \
    libopenblas-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# Install Claude Code globally
# Using --no-fund and --no-audit flags to reduce installation noise
RUN npm install -g @anthropic-ai/claude-code --no-fund --no-audit

# Set the working directory
WORKDIR /workspace

# Verify Claude Code installation
RUN claude --version

# Pre-build a virtualenv with prodiphy's runtime and dev dependencies (pytest,
# pytest-cov, ruff) so the container is ready for development as soon as it
# starts. The editable install is refreshed against the live-mounted
# workspace by the devcontainer's postCreateCommand.
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install -e ".[dev]"
ENV PATH="/opt/venv/bin:${PATH}"

# Keep container running for devcontainer usage
CMD ["sleep", "infinity"]