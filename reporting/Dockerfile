FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Create the reporting directory
RUN mkdir -p /app/reporting

COPY . .

# Expose the FastAPI and Streamlit ports
EXPOSE 8000
EXPOSE 8501

# Run services sequentially
CMD ["sh", "-c", "python project.py && streamlit run api.py --server.port=8501 --server.address=0.0.0.0 & uvicorn api:app --host 0.0.0.0 --port 8000"]