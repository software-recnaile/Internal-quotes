# Use official Python runtime as base image
FROM python:3.11-slim

# Set working directory
WORKDIR /Backend

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY ./Backend ./Backend

# Expose port
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "Backend.main:app", "--host", "0.0.0.0", "--port", "8000"]