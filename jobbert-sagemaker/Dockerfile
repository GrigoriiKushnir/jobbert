FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1

COPY requirements.txt /opt/ml/code/requirements.txt
RUN pip install --no-cache-dir -r /opt/ml/code/requirements.txt

COPY serve.py /opt/ml/code/serve.py

WORKDIR /opt/ml/code

RUN echo '#!/bin/bash' > /opt/ml/code/serve && \
    echo 'uvicorn serve:app --host 0.0.0.0 --port 8080' >> /opt/ml/code/serve && \
    chmod +x /opt/ml/code/serve

# SageMaker will call this with 'serve' as an argument
ENV PATH="/opt/ml/code:${PATH}"

ENTRYPOINT ["serve"]
