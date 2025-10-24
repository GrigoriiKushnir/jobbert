import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

try:
    model = SentenceTransformer('TechWolf/JobBERT-v3')
    logger.info('Model loaded successfully')
except Exception as e:
    logger.error(f'Failed to load model: {e}')
    raise


class InferenceRequest(BaseModel):
    inputs: list[str]


@app.get('/ping')
def ping():
    return {'status': 'healthy'}


@app.post('/invocations')
def invocations(req: InferenceRequest):
    try:
        texts = req.inputs
        if not texts:
            raise HTTPException(status_code=400, detail='No input texts provided')

        embeddings = model.encode(texts)
        return embeddings.tolist()
    except Exception as e:
        logger.error(f'Inference error: {e}')
        raise HTTPException(status_code=500, detail=str(e))
