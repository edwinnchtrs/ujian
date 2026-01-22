# Use official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Create static folders if they don't exist
RUN mkdir -p static/plots

# Expose port 5000
EXPOSE 7860

# Run the app (Hugging Face uses 7860 as default)
CMD ["python", "app.py"]
