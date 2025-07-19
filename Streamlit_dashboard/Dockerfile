# Use python image as base
FROM python:3.11-slim

# Install system tools
RUN apt-get update && apt-get install -y \
    nano \
    unzip \
    curl \
    build-essential \
    gfortran \
    && apt-get clean 

# Create and switch to non-root user (required by Hugging Face)
RUN useradd -m -u 1000 user
USER user

# Set up environment variables and working directory
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app

# Install Python dependencies
COPY --chown=user requirements.txt $HOME/app/
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy rest of the app code
COPY --chown=user . $HOME/app

# Expose the default port used by Hugging Face
EXPOSE 7860

# Start Streamlit app
CMD ["streamlit", "run", "--server.port=7860", "app.py", "--server.headless=true"]