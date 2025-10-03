
# AI_OCR

AI_OCR is a receipt data extraction service built with NestJS, leveraging BERT for natural language understanding and Tesseract for optical character recognition (OCR). The BERT model is run using ONNX for fast inference, and you must download and convert a BERT model from Hugging Face to ONNX format before use.

## Features
- Extracts and formats data from receipt images
- Uses Tesseract.js for OCR
- Uses BERT (via ONNX) for line classification and item extraction
- Custom tokenizer for BERT

## Requirements
- Node.js (recommended v16+)
- Python (for ONNX conversion)
- NestJS
- Tesseract.js
- onnxruntime-node
- sharp

## Setup

### 1. Clone the repository
```powershell
# Clone the repo
git clone https://github.com/zayarmoekaung/AI_OCR.git
cd AI_OCR
```

### 2. Install dependencies
```powershell
npm install
```

### 3. Download and convert BERT model

You need to download the BERT base model from Hugging Face and convert it to ONNX format. Place the files in `models/bert-base-uncased/`:

- `model.onnx`
- `vocab.txt`
- `config.json`

#### Steps:
1. Download the BERT base uncased model from [Hugging Face](https://huggingface.co/bert-base-uncased).
2. Use the [transformers](https://github.com/huggingface/transformers) and [onnxruntime-tools](https://github.com/microsoft/onnxruntime) Python packages to convert the model:

```python
from transformers import BertModel
from transformers import BertTokenizer
from transformers.onnx import export
import torch

model = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Export to ONNX
export(
	preprocessor=tokenizer,
	model=model,
	config=model.config,
	opset=11,
	output='model.onnx',
	tokenizer=tokenizer
)
```

Or use the official Hugging Face ONNX export tool:
```bash
transformers-cli env
python -m transformers.onnx --model=bert-base-uncased --feature=sequence-classification ./model.onnx
```

3. Copy `model.onnx`, `vocab.txt`, and `config.json` to `models/bert-base-uncased/`.

### 4. Run the service
```powershell
npm run start
```

## Usage
Send a receipt image to the API endpoint. The service will:
- Preprocess the image
- Perform OCR using Tesseract
- Classify and extract items using BERT (ONNX)

## Project Structure
- `src/receipt/receipt.service.ts`: Main logic for OCR and BERT inference
- `src/util/tokenizer.ts`: Custom BERT tokenizer
- `models/bert-base-uncased/`: Place your ONNX model and tokenizer files here

## License
MIT

