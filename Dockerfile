# Use a base Python image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy project files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port (default for FastAPI or Flask is 8000)
EXPOSE 8000

# Command to run the app
CMD ["python", "email_spam.py"]
