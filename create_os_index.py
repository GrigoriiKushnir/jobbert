import json

import boto3
import dotenv
import pandas as pd
from botocore.config import Config
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

BOTO3_CONFIG = Config(
    retries={'max_attempts': 3, 'mode': 'standard'},
    max_pool_connections=200
)


def create_os_index():
    OS_CLIENT.indices.create(
        index='jobtitles',
        body={
            'settings': {
                'number_of_shards': 1,
                'number_of_replicas': 0,
                'index': {
                    'knn': True,
                }
            },
            'mappings': {
                'properties': {
                    'job_title': {'type': 'keyword'},
                    'embedding_vector': {
                        "type": "knn_vector",
                        "dimension": 1024,
                        "space_type": "cosinesimil",
                        "method": {
                            "name": "hnsw",
                            "engine": "lucene",
                            "parameters": {
                                "encoder": {
                                    "name": "sq"
                                },
                                "m": 48,
                                "ef_construction": 512
                            }
                        }
                    }
                }
            }
        }
    )


def generate_embedding_from_text(text):
    boto3_client = boto3.client(
        'sagemaker-runtime',
        region_name='eu-west-3',
        config=BOTO3_CONFIG
    )
    payload = {'inputs': [text]}
    response = boto3_client.invoke_endpoint(
        EndpointName=settings.SAGEMAKER_ENDPOINT_NAME,
        ContentType='application/json',
        Body=json.dumps(payload)
    )
    embeddings = json.loads(response['Body'].read())
    return embeddings[0]

def write_to_os():
    df = pd.read_parquet('curated_job_titles_embeddings.parquet')
    total_titles = len(df)
    print('Total titles: {}'.format(total_titles))
    for index, row in df.iterrows():
        print(f'Index: {index}/{total_titles}')
        OS_CLIENT.index(
            index='jobtitles',
            body={
                'job_title': row['job_title'],
                'embedding_vector': generate_embedding_from_text(row['job_title'])
            }
        )



def call():
    write_to_os()
    # create_os_index()


if __name__ == '__main__':
    call()

