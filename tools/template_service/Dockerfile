FROM python:3.11-slim

WORKDIR /service
COPY . /service

# Install runtime dependencies if a requirements file is provided
RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi

CMD ["python", "service.py"]
