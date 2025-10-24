import json

import boto3
import dotenv
from opensearchpy import OpenSearch, AWSV4SignerAuth, RequestsHttpConnection

import settings

dotenv.load_dotenv()

OS_CLIENT = OpenSearch(
    hosts=[{'host': settings.OS_DOMAIN_HOST, 'port': settings.OS_DOMAIN_PORT}],
    timeout=30,
    http_auth=AWSV4SignerAuth(
        boto3.Session().get_credentials(),
        settings.AWS_REGION
    ),
    connection_class=RequestsHttpConnection,
    headers={'Host': 'place-holder-host-xyz.eu-west-3.es.amazonaws.com'}
)


def create_connector() -> str:
    response = OS_CLIENT.transport.perform_request(
        'POST',
        '/_plugins/_ml/connectors/_create',
        body={
            'name': 'sagemaker: embedding',
            'description': 'Connector for Sagemaker JobBertV3 embedding model',
            'version': 1,
            'protocol': 'aws_sigv4',
            'credential': {
                'roleArn': settings.AWS_ROLE_ARN
            },
            'parameters': {
                'region': 'eu-west-3',
                'service_name': 'sagemaker'
            },
            'actions': [
                {
                    'action_type': 'predict',
                    'method': 'POST',
                    'headers': {
                        'content-type': 'application/json'
                    },
                    'url': settings.SAGEMAKER_ENDPOINT_URL,
                    'request_body': '{ "inputs": "${parameters.text}" }'
                }
            ]
        }
    )
    return response['connector_id']


def get_connectors():
    response = OS_CLIENT.transport.perform_request(
        'POST',
        '/_plugins/_ml/connectors/_search',
        body={
            'query': {
                'term': {
                    'parameters.service_name': 'sagemaker'
                }
            }
        }
    )
    return response['hits']['hits']


def cleanup_connectors(sagemaker_connectors: list[dict]):
    for sagemaker_connector in sagemaker_connectors:
        print(f'Cleaning up connector {sagemaker_connector["_id"]}')
        OS_CLIENT.transport.perform_request(
            'DELETE',
            f'/_plugins/_ml/connectors/{sagemaker_connector["_id"]}'
        )


def get_models(sagemaker_connectors: list[dict]):
    connector_ids = [connector['_id'] for connector in sagemaker_connectors]
    response = OS_CLIENT.transport.perform_request(
        'POST',
        '/_plugins/_ml/models/_search',
        body={
            'query': {
                'terms': {
                    'connector_id.keyword': connector_ids
                }
            }
        }
    )
    return response['hits']['hits']


def cleanup_models(sagemaker_models: list[dict]):
    for sagemaker_model in sagemaker_models:
        print(f'Cleaning up model {sagemaker_model["_id"]}')
        if sagemaker_model['_source']['model_state'] == 'DEPLOYED':
            OS_CLIENT.transport.perform_request(
                'POST',
                f'/_plugins/_ml/models/{sagemaker_model["_id"]}/_undeploy'
            )
        OS_CLIENT.transport.perform_request(
            'DELETE',
            f'/_plugins/_ml/models/{sagemaker_model["_id"]}'
        )


def register_model(sagemaker_connector_id: str):
    response = OS_CLIENT.transport.perform_request(
        'POST',
        '/_plugins/_ml/models/_register',
        body={
            'name': 'jobbertv3',
            'function_name': 'remote',
            'connector_id': sagemaker_connector_id,
            'model_config': {
                'model_type': 'bert',
                'embedding_dimension': 1024,
                'framework_type': 'SENTENCE_TRANSFORMERS'
            }
        }
    )
    return response['model_id']


def call():
    # Create connector
    sagemaker_connectors = get_connectors()
    sagemaker_models = get_models(sagemaker_connectors)
    if sagemaker_connectors:
        print('\u2705 Connector already exists:')
        print(json.dumps(sagemaker_connectors, indent=2))
    else:
        connector_id = create_connector()
        print(f'\u2705 Created connector: {connector_id}')

    if len(sagemaker_connectors) > 1:
        cleanup_models(sagemaker_models)
        cleanup_connectors(sagemaker_connectors)

    # Register model
    if sagemaker_models:
        print('\u2705 Model already exists:')
        print(json.dumps(sagemaker_models, indent=2))
        model_id = sagemaker_models[0]['_id']
    else:
        model_id = register_model(sagemaker_connector_id=sagemaker_connectors[0]['_id'])
        print(f'\u2705 Registered model: {model_id}')

    if len(sagemaker_models) > 1:
        cleanup_models(sagemaker_models)

    # Deploy model
    OS_CLIENT.transport.perform_request(
        'POST',
        f'/_plugins/_ml/models/{model_id}/_deploy'
    )
    print(f'\u2705 Deployed model: {model_id}')
    print(json.dumps(sagemaker_models[0], indent=2))


if __name__ == '__main__':
    call()

# (.venv) ➜  jobbert python3 os_query.py
# Status: 200
# Response: {'connector_id':'dnNuBpoBitaEDxadwht8'}
