import dotenv
import sagemaker

import settings

dotenv.load_dotenv()

SAGEMAKER_ROLE = 'arn:aws:iam::080274686453:role/sagemaker_admin'


def call():
    hf_model = sagemaker.Model(
        role=SAGEMAKER_ROLE,
        image_uri='080274686453.dkr.ecr.eu-west-3.amazonaws.com/dev/jobbertv3:latest',
    )

    try:
        hf_model.deploy(
            endpoint_name=settings.SAGEMAKER_ENDPOINT_NAME,
            instance_type=settings.HF_MODEL_INSTANCE_TYPE,
            initial_instance_count=1,
        )
    except Exception as e:
        print(e)


if __name__ == '__main__':
    call()
