import { Injectable } from '@nestjs/common';
import { createWorker } from 'tesseract.js';
import * as sharp from 'sharp';
import { InferenceSession, Tensor } from 'onnxruntime-node';
import { Receipt, Item } from 'src/types/receipt.type';
import * as path from 'path';
import { BertTokenizer } from 'src/util/tokenizer';

@Injectable()
export class ReceiptService {
    private session: InferenceSession | null = null;
    private tokenizer: BertTokenizer;

    constructor() {
        const vocabPath = path.join(__dirname, '../../models/bert-base-uncased/vocab.txt');
        const configPath = path.join(__dirname, '../../models/bert-base-uncased/config.json');
        this.tokenizer = new BertTokenizer(vocabPath, configPath);
        this.loadModel().catch(error => {
            console.error('Error loading ONNX model:', error);
        });
    }
    private async loadModel(): Promise<void> {
        try {
            const modelPath = path.join(__dirname, '../../models/bert-base-uncased/model.onnx');
            this.session = await InferenceSession.create(modelPath);
        } catch (error) {
            throw new Error(`Failed to load ONNX model: ${error.message}`);
        }
    }
    async extractAndFormatData(file: Express.Multer.File): Promise<Receipt> {
    if (!['image/jpeg', 'image/png'].includes(file.mimetype)) {
      throw new Error('Invalid file format. Only JPEG and PNG are supported.');
    }

    const processedImage = await this.preprocessImage(file.buffer);
    const text = await this.performOCR(processedImage);
    return this.formatReceipt(text);
  }

  private async preprocessImage(buffer: Buffer): Promise<Buffer> {
    // Enhanced preprocessing for better OCR accuracy
    return sharp(buffer)
      .grayscale()
      .normalize() // Normalize brightness and contrast
      .sharpen() // Sharpen edges for better text recognition
      .threshold(80) // Increase contrast
      .resize(2000, null, { fit: 'inside' }) // Upscale for clarity
      .toBuffer();
  }

  private async performOCR(buffer: Buffer): Promise<string> {
    const worker = await createWorker('eng');
    try {
      const { data: { text } } = await worker.recognize(buffer);
      return text.trim().replace(/\s+/g, ' '); // Normalize whitespace
    } catch (error) {
      throw new Error(`OCR failed: ${error.message}`);
    } finally {
      await worker.terminate();
    }
  }

  private async formatReceipt(text: string): Promise<Receipt> {
    const data: Receipt = { merchant: '', date: '', total: 0, items: [] };
    const lines = text.split('\n').filter(line => line.trim());

    const labels = await this.classifyLines(lines);
    const classifiedLines = lines.map((line, idx) => ({ line, label: labels[idx] }));

    // Extract merchant (first non-item line or store name)
    const merchantLine = classifiedLines.find(l => l.label === 'merchant') || classifiedLines.find(l => l.line.includes('STOP & SHOP'));
    if (merchantLine) data.merchant = merchantLine.line.split('-')[0].trim(); // Take part before address

    // Extract date (look for date format)
    const dateLine = classifiedLines.find(l => l.label === 'date' || /\d{2}\/\d{2}\/\d{2}.*\d{2}:\d{2}.*/.test(l.line));
    if (dateLine) data.date = dateLine.line.match(/\d{2}\/\d{2}\/\d{2}.*\d{2}:\d{2}.*/)![0];

    // Extract total (look for balance or total-like lines)
    const totalLine = classifiedLines.find(l => l.label === 'total' || l.line.includes('BALANCE') || l.line.includes('$'));
    if (totalLine) {
      const totalMatch = totalLine.line.match(/\$?(\d+\.\d{2})/);
      if (totalMatch) data.total = parseFloat(totalMatch[1]);
    }

    // Extract items (group by price lines)
    let currentItem = { name: '', price: 0, quantity: 1 };
    for (const { line, label } of classifiedLines) {
      if (label === 'item') {
        const priceMatch = line.match(/\$?(\d+\.\d{2})/);
        if (priceMatch) {
          currentItem.price = parseFloat(priceMatch[1]);
          data.items.push({ ...currentItem });
          currentItem = { name: '', price: 0, quantity: 1 };
        } else {
          currentItem.name = line.trim() || currentItem.name || 'Unknown Item';
        }
      }
    }

    return data;
  }

  private async classifyLines(lines: string[]): Promise<string[]> {
    if (!this.session) {
      throw new Error('ONNX model not initialized');
    }

    const labels: string[] = [];
    const labelMap = ['other', 'merchant', 'date', 'total', 'item'];

    for (const line of lines) {
      const { inputIds, attentionMask, tokenTypeIds } = this.tokenizer.tokenize(line);

      const inputTensor = new Tensor('int64', BigInt64Array.from(inputIds.map(BigInt)), [1, this.tokenizer['maxLength']]);
      const attentionMaskTensor = new Tensor('int64', BigInt64Array.from(attentionMask.map(BigInt)), [1, this.tokenizer['maxLength']]);
      const tokenTypeIdsTensor = new Tensor('int64', BigInt64Array.from(tokenTypeIds.map(BigInt)), [1, this.tokenizer['maxLength']]);
      const feeds = {
        input_ids: inputTensor,
        attention_mask: attentionMaskTensor,
        token_type_ids: tokenTypeIdsTensor,
      };

      try {
        const results = await this.session.run(feeds);
        const output = results.logits.data as Float32Array;

        // Aggregate logits across non-special tokens
        const numTokens = inputIds.indexOf(this.tokenizer['specialTokens']['[SEP]']);
        let maxScore = -Infinity;
        let labelIdx = 0;
        for (let i = 1; i < numTokens; i++) {
          const logits = output.slice(i * labelMap.length, (i + 1) * labelMap.length);
          const maxLogitIdx = logits.reduce((maxIdx, val, idx, arr) => val > arr[maxIdx] ? idx : maxIdx, 0);
          if (logits[maxLogitIdx] > maxScore) {
            maxScore = logits[maxLogitIdx];
            labelIdx = maxLogitIdx;
          }
        }
        labels.push(labelMap[labelIdx]);
      } catch (error) {
        console.error(`Inference failed for line "${line}": ${error.message}`);
        labels.push('other');
      }
    }

    return labels;
  }
    private async classifyItem(line: string): Promise<Item> {
        const item: Item = { name: '', price: 0, quantity: 1 };
        const parts = line.split(/\s+/);
        if (parts.length < 2) {
            return item; // Not enough parts to classify
        }

        // Assume last part is price, and the rest is the name
        const pricePart = parts.pop();
        if (pricePart && /^\d+(\.\d{2})?$/.test(pricePart)) {
            item.price = parseFloat(pricePart);
            item.name = parts.join(' ');
        } else {
            item.name = line; // If no valid price found, treat whole line as name
        }

        return item;
    }
}
