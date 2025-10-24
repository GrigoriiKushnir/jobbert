FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1

COPY requirements.txt /opt/ml/code/requirements.txt
RUN pip install --no-cache-dir -r /opt/ml/code/requirements.txt
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('GrishaKushnir/jobbert-onnx', backend='onnx')"

COPY serve.py /opt/ml/code/serve.py

WORKDIR /opt/ml/code
RUN echo '#!/bin/bash' > /opt/ml/code/serve && \
    echo 'gunicorn serve:app -w 2 --bind 0.0.0.0:8080 --timeout 600 -k uvicorn.workers.UvicornWorker' >> /opt/ml/code/serve && \
    chmod +x /opt/ml/code/serve

# SageMaker will call this with 'serve' as an argument
ENV PATH="/opt/ml/code:${PATH}"

ENTRYPOINT ["serve"]
