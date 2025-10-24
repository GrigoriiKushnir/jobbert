AWS_REGION = 'eu-west-3'
AWS_SERVICE = 'es'
AWS_ROLE_ARN = 'arn:aws:iam::080274686453:role/sagemaker_admin'

HF_MODEL_INSTANCE_TYPE = 'ml.c5.xlarge'
SAGEMAKER_ENDPOINT_NAME = f"jobbert3-{HF_MODEL_INSTANCE_TYPE.replace('.', '-')}-1"

OS_DOMAIN_HOST = 'localhost'
OS_DOMAIN_PORT = 9201
OS_DOMAIN_ENDPOINT = f'http://{OS_DOMAIN_PORT}:{OS_DOMAIN_PORT}'

SAGEMAKER_ENDPOINT_URL = f'https://runtime.sagemaker.eu-west-3.amazonaws.com/endpoints/{SAGEMAKER_ENDPOINT_NAME}/invocations'
