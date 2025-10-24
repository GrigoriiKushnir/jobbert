import dotenv
from botocore.exceptions import ClientError
from sagemaker.huggingface import HuggingFaceModel
from sagemaker.huggingface import get_huggingface_llm_image_uri

import settings

dotenv.load_dotenv()

SAGEMAKER_ROLE = 'arn:aws:iam::080274686453:role/sagemaker_admin'
HF_MODEL_CONFIG = {
    'HF_MODEL_ID': 'GrishaKushnir/jobbert-onnx',
}


def call():
    hf_model = HuggingFaceModel(
        role=SAGEMAKER_ROLE,
        image_uri=get_huggingface_llm_image_uri('huggingface-tei-cpu'),
        env=HF_MODEL_CONFIG
    )

    try:
        hf_model.deploy(
            endpoint_name=settings.SAGEMAKER_ENDPOINT_NAME,
            instance_type=settings.HF_MODEL_INSTANCE_TYPE,
            initial_instance_count=1,
        )
    except ClientError as e:
        if 'Cannot create already existing endpoint' in e.response['Error']['Message']:
            print(f'Endpoint {settings.SAGEMAKER_ENDPOINT_NAME} already exists')


if __name__ == '__main__':
    call()
