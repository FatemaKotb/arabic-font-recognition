# Use the official Python base image
FROM python:3.11.5

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file to the working directory
COPY requirements.txt .

# Install the Python dependencies
RUN pip install -r requirements.txt

RUN pip3 install opencv-python-headless

# Copy the application code to the working directory
COPY . .

# Expose the port on which the application will run
EXPOSE 8080

# Run the FastAPI application using uvicorn server
CMD ["uvicorn", "mlapi:app", "--host", "0.0.0.0", "--port", "8080"]