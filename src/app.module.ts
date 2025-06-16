import { Module } from '@nestjs/common';
import { AppController } from './app.controller';
import { AppService } from './app.service';
import { ReceiptModule } from './receipt/receipt.module';
import { MulterModule } from '@nestjs/platform-express';

@Module({
  imports: [ReceiptModule,
     MulterModule.register({
      dest: './uploads',
    }),
  ],
  controllers: [AppController],
  providers: [AppService],
})
export class AppModule {}
