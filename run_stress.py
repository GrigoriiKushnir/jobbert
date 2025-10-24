import json
import time

import boto3
import dotenv
from botocore.config import Config
from locust import User
from locust import constant_pacing
from locust import events
from locust import task

import settings

dotenv.load_dotenv()
BOTO3_CONFIG = Config(
    retries={'max_attempts': 3, 'mode': 'standard'},
    max_pool_connections=200
)
BOTO3_CLIENT = boto3.client(
    'sagemaker-runtime',
    region_name=settings.AWS_REGION,
    config=BOTO3_CONFIG,
)


class SageMakerUser(User):
    host = 'https://sagemaker'  # dummy

    wait_time = constant_pacing(1.0)  # 1 req/sec per user

    @task
    def query_jobbert(self):
        payload = {'inputs': 'today is rainy'}
        start_time = time.time()
        try:
            response = BOTO3_CLIENT.invoke_endpoint(
                EndpointName=settings.SAGEMAKER_ENDPOINT_NAME,
                ContentType='application/json',
                Body=json.dumps(payload)
            )
            result = json.loads(response['Body'].read())

            total_time = int((time.time() - start_time) * 1000)  # ms
            events.request.fire(
                request_type='sagemaker',
                name='invoke_endpoint',
                response_time=total_time,
                response_length=len(json.dumps(result)),
                exception=None
            )
        except Exception as e:
            total_time = int((time.time() - start_time) * 1000)
            events.request.fire(
                request_type='sagemaker',
                name='invoke_endpoint',
                response_time=total_time,
                response_length=0,
                exception=e
            )


'''
locust -f stress.py -u 1 -r 10 --run-time 10m --stop-timeout 10
'''
