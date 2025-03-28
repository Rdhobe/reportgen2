# Use the official Python image as a base
FROM python:3.11

# Set the working directory inside the container
WORKDIR /app

# Copy the application files to the container
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port your Flask app runs on
EXPOSE 8000

# Command to run the Flask application
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8000", "app:app"]
