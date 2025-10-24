from sentence_transformers import SentenceTransformer
from sentence_transformers import export_dynamic_quantized_onnx_model
from sentence_transformers import export_optimized_onnx_model

MODEL_ID = 'TechWolf/JobBERT-v3'
OUTPUT_DIR = './jobbert_onnx'


def call():
    onnx_model = SentenceTransformer(
        MODEL_ID,
        backend='onnx',
        model_kwargs={'export': True}
    )
    onnx_model.save_pretrained(OUTPUT_DIR)

    for optimization_config in ['O1', 'O2', 'O3', 'O4']:
        export_optimized_onnx_model(
            onnx_model,
            optimization_config=optimization_config,
            model_name_or_path=OUTPUT_DIR,
        )

    for quantization_config in ['arm64', 'avx2', 'avx512', 'avx512_vnni']:
        export_dynamic_quantized_onnx_model(
            onnx_model,
            quantization_config=quantization_config,
            model_name_or_path=OUTPUT_DIR,
        )

    openvino_model = SentenceTransformer(MODEL_ID, backend='openvino')
    openvino_model.save_pretrained(OUTPUT_DIR)


if __name__ == '__main__':
    call()
