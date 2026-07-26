# Dockerfile
# ----------
# WHY: Docker lets any student run this app with the exact same environment,
#      without worrying about "it works on my machine" problems.
# WHAT: Builds a container image with Python 3.12, installs our dependencies,
#       copies the source code in, and starts the Streamlit server.

FROM python:3.12-slim

# Prevent Python from writing .pyc files and buffer-less logs, so
# "docker logs" shows output immediately (helpful for debugging in class).
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory inside the container.
WORKDIR /app

# Install OS-level build tools needed by faiss-cpu / pandas wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only the requirements file first. Docker caches this layer, so
# rebuilding after a code change (not a dependency change) is much faster.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the rest of the project source code.
COPY . .

# Make sure the folders the app writes to actually exist.
RUN mkdir -p uploads reports

# Streamlit's default port.
EXPOSE 8501

# Run Streamlit. --server.address=0.0.0.0 lets you reach it from outside the container.
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
