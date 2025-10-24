import json

import boto3
import dotenv
import numpy as np
import requests
from botocore.config import Config
from sentence_transformers import SentenceTransformer

import settings

np.set_printoptions(threshold=np.inf)

dotenv.load_dotenv()

BOTO3_CONFIG = Config(
    retries={'max_attempts': 3, 'mode': 'standard'},
    max_pool_connections=200
)


def call_using_endpoint():
    boto3_client = boto3.client(
        'sagemaker-runtime',
        region_name='eu-west-3',
        config=BOTO3_CONFIG
    )
    payload = {'inputs': f'today is rainy'}
    response = boto3_client.invoke_endpoint(
        EndpointName=settings.SAGEMAKER_ENDPOINT_NAME,
        ContentType='application/json',
        Body=json.dumps(payload)
    )
    embeddings = json.loads(response['Body'].read())
    print(f'Sagemaker model embeddings length: {len(embeddings[0])}')
    # print(f'Sagemaker model embeddings {embeddings[0]}')


def call_using_hf_model():
    model = SentenceTransformer(
        'GrishaKushnir/jobbert-onnx',
        backend='onnx',
        model_kwargs={
            'file_name': 'onnx/model.onnx'
        }

    )
    sentences = ['today is rainy']
    embeddings = model.encode(sentences)
    print(f'HF model embeddings length: {len(embeddings[0])}')
    # print(f'HF model embeddings {embeddings[0]}')


def call_using_local_model():
    response = requests.post(
        'http://127.0.0.1:8080/embed',
        json={'inputs': 'today is rainy'},
        headers={'Content-Type': 'application/json'}
    )
    embeddings = response.json()
    print(f'Local model embeddings length: {len(embeddings[0])}')


def call():
    # call_using_endpoint()
    # call_using_hf_model()
    call_using_local_model()


if __name__ == '__main__':
    call()
