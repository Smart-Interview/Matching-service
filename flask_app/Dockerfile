# Use an official Python runtime as a base image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

RUN python -m nltk.downloader wordnet -d /usr/local/share/nltk_data \
    && python -m nltk.downloader omw-1.4 -d /usr/local/share/nltk_data

# Expose port 5000 (the default Flask port)
EXPOSE 5000

# Set environment variables (if you have environment variables in .env file)
# These can be overridden in the docker-compose.yml if you use it later
ENV FLASK_ENV=development
ENV FLASK_APP=app.py

# Run the Flask app
CMD ["flask", "run", "--host=0.0.0.0"]
