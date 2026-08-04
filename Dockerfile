# Using slim image for reduced image size
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

# Using extra pip install for non-gpu version of torch, reducing image size
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir torch \
       --index-url https://download.pytorch.org/whl/cpu

COPY app.py .
COPY src ./src
COPY models ./models

EXPOSE 8080

CMD ["python", "app.py"]