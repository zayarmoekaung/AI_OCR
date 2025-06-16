import { Controller, Post, UploadedFile, UseInterceptors } from '@nestjs/common';
import { FileInterceptor } from '@nestjs/platform-express';
import { ReceiptService } from './receipt.service';
import { Receipt } from 'src/types/receipt.type';
@Controller('receipt')
export class ReceiptController {
  constructor(private readonly receiptService: ReceiptService) {}

  @Post('extract')
  @UseInterceptors(FileInterceptor('file'))
  async extract(@UploadedFile() file: Express.Multer.File): Promise<Receipt> {
    if (!file) {
      throw new Error('No file uploaded');
    }
    return this.receiptService.extractAndFormatData(file);
  }
}