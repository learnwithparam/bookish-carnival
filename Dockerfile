# Use a Python image with uv pre-installed
FROM ghcr.io/astral-sh/uv:python3.10-bookworm-slim

# Set working directory
WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy project files
COPY pyproject.toml .
COPY . .

# Install dependencies
RUN uv sync --frozen

# Expose port
EXPOSE 8000

# Run the app
CMD ["uv", "run", "chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "8000"]
