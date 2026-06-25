# Adapter r=16

Adapter này được train từ `unsloth/Qwen2.5-3B-bnb-4bit` trên 200 samples
của dataset `5CD-AI/Vietnamese-alpaca-gpt4-gg-translated` (180 train + 20 eval).

## Cấu hình

- **Rank**: 16
- **Alpha**: 32
- **Target modules**: `["q_proj", "v_proj"]`
- **Dropout**: 0
- **Epochs**: 3
- **Effective batch size**: 8 (batch=1, grad_accum=8)
- **Learning rate**: 2e-4 (cosine)
- **Trainable params**: 3,686,400 (0.12% of 3,089,625,088)
- **Training time**: 4.26 phút trên Tesla T4
- **Peak VRAM**: 6.62 GB
- **Eval loss**: 1.5161
- **Perplexity**: 4.55

## Files

- `adapter_model.safetensors` — LoRA weights (B × A matrices cho q_proj và v_proj của 36 layers)
- `adapter_config.json` — PEFT config

## Reproduce

Sau khi train xong trong notebook (cell 16), adapter đã được save tự động
vào `OUTPUT_DIR/r16/`. Để load lại:

```python
from peft import PeftModel
from unsloth import FastLanguageModel

base, tok = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-3B-bnb-4bit",
    max_seq_length=1024,
    load_in_4bit=True,
)
model = PeftModel.from_pretrained(base, "adapters/r16")
FastLanguageModel.for_inference(model)
```