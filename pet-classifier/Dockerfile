# Use official miniconda image
FROM continuumio/miniconda3:latest

# Set working directory
WORKDIR /app

# Copy app files
COPY ./app /app
COPY requirements.txt /app/requirements.txt

# Create conda environment and install packages
RUN conda create -y -n petenv python=3.11 \
    && conda run -n petenv pip install --upgrade pip \
    && conda run -n petenv conda install -y pytorch torchvision torchaudio cpuonly -c pytorch \
    && conda run -n petenv pip install -r requirements.txt

# Make sure the environment is activated when container runs
ENV PATH=/opt/conda/envs/petenv/bin:$PATH

# Expose Flask port
EXPOSE 5000

# Run the Flask app
CMD ["python", "app.py"]
