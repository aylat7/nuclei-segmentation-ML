FROM python:3.10-slim

# System dependencies for image processing and OpenCV compatibility
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies before copying source for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source
COPY . .

# Create outputs directory so it exists on first run
RUN mkdir -p outputs

EXPOSE 7860

CMD ["python", "app.py"]
