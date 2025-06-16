import * as fs from 'fs';
import * as path from 'path';

export class BertTokenizer {
    private vocab: { [word: string]: number };
    private maxLength: number;
    private doLowerCase: boolean;
    private specialTokens: { [key: string]: number };

    constructor(vocabPath: string, configPath: string) {
        // Load tokenizer configuration
        const tokenizerConfig = JSON.parse(fs.readFileSync(path.join(path.dirname(vocabPath), 'tokenizer.json'), 'utf-8'));
        const modelConfig = JSON.parse(fs.readFileSync(configPath, 'utf-8'));

        // Load vocabulary
        const vocabContent = fs.readFileSync(vocabPath, 'utf-8');
        this.vocab = vocabContent
            .split('\n')
            .reduce((acc, word, idx) => ({ ...acc, [word.trim()]: idx }), {});

        // Load settings from config.json
        this.maxLength = modelConfig.max_position_embeddings || 512;
        this.doLowerCase = tokenizerConfig.do_lower_case || true;
        this.specialTokens = {
            '[CLS]': this.vocab['[CLS]'] || 101,
            '[SEP]': this.vocab['[SEP]'] || 102,
            '[PAD]': this.vocab['[PAD]'] || 0,
            '[UNK]': this.vocab['[UNK]'] || 100,
        };
    }

    tokenize(text: string): { inputIds: number[], attentionMask: number[], tokenTypeIds: number[] } {
        if (this.doLowerCase) {
            text = text.toLowerCase();
        }

        // Split into words and apply WordPiece tokenization
        let tokens: string[] = ['[CLS]'];
        const words = text.split(/\s+/);
        for (const word of words) {
            const subwords = this.wordpieceTokenize(word);
            tokens = tokens.concat(subwords);
            if (tokens.length >= this.maxLength - 1) break;
        }
        tokens.push('[SEP]');

        // Convert to input IDs
        const inputIds = tokens.map(token => this.vocab[token] || this.specialTokens['[UNK]']);
        const attentionMask = inputIds.map(() => 1);
        const tokenTypeIds = inputIds.map(() => 0);

        // Pad to maxLength
        while (inputIds.length < this.maxLength) {
            inputIds.push(this.specialTokens['[PAD]']);
            attentionMask.push(0);
            tokenTypeIds.push(0);
        }

        return { inputIds, attentionMask, tokenTypeIds };
    }

    private wordpieceTokenize(word: string): string[] {
        const maxWordLength = 200; // Prevent infinite loops
        if (word.length > maxWordLength) {
            return ['[UNK]'];
        }

        const tokens: string[] = [];
        let currentWord = word;
        while (currentWord) {
            let found = false;
            let subword = currentWord;

            // Try to find the longest matching subword
            while (subword && subword.length > 0) {
                const token = tokens.length > 0 ? '##' + subword : subword;
                if (this.vocab[token]) {
                    tokens.push(token);
                    currentWord = currentWord.slice(subword.length);
                    found = true;
                    break;
                }
                subword = subword.slice(0, -1);
            }

            if (!found) {
                tokens.push('[UNK]');
                break;
            }
        }

        return tokens;
    }
}