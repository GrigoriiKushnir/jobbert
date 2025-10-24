1. Create IAM Role:

sagemaker_admin
AmazonS3FullAccess
AmazonSageMakerFullAccess

Trust Policy
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": [
          "sagemaker.amazonaws.com",
          "es.eu-west-3.amazonaws.com"
        ]
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
```

2. Allow running model on all nodes:

```
PUT _cluster/settings
{
  "persistent": {
    "plugins.ml_commons.only_run_on_ml_node": false
  }
}
```

3. Create ONNX model and upload to HF
3. Run `create_api.py` to deploy the HF hosted JobBertV3 model on Sagemaker
4. Run `create_os_connection.py` to create the connector from OS to Sagemaker model


# Run locally
export DOCKER_DEFAULT_PLATFORM=linux/amd64
model=GrishaKushnir/jobbert-onnx

volume=$PWD/data # share a volume with the Docker container to avoid downloading weights every run

docker run -p 8080:80 -v $volume:/data --pull always ghcr.io/huggingface/text-embeddings-inference:1.8 --model-id $model
