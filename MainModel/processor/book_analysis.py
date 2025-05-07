import json
import torch
import math
import evaluate
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset

class DeepSeekBookProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.book_data = self.load_book_data()

    def load_book_data(self):
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print("❌ Dosya bulunamadı.")
            return None
        except json.JSONDecodeError:
            print("❌ JSON dosyası hatalı.")
            return None

    def explore_book(self):
        if not self.book_data:
            return "Veri yüklenemedi."

        return {
            'kitap_adi': self.book_data.get('kitap_adi', 'Bilinmiyor'),
            'sayfa_sayisi': len(self.book_data.get('sayfalar', []))
        }

    def get_full_text(self):
        if not self.book_data:
            return ""
        pages = self.book_data.get('sayfalar', [])
        full_text = "\n".join(p.get("icerik", "") for p in pages)
        return full_text.strip()

    def count_words(self):
        return len(self.get_full_text().split())

    def count_sentences(self):
        return len([s for s in self.get_full_text().split('.') if s.strip()])

    def count_paragraphs(self):
        return len([p for p in self.get_full_text().split('\n\n') if p.strip()])

    def split_into_chunks(self, max_words=750):
        text = self.get_full_text()
        words = text.split()
        chunks = []
        for i in range(0, len(words), max_words):
            chunk = words[i:i + max_words]
            chunks.append(" ".join(chunk))
        return chunks

    def export_jsonl(self, output_path, chunks):
        with open(output_path, 'w', encoding='utf-8') as f:
            for chunk in chunks:
                if chunk.strip():
                    json.dump({"output": chunk.strip()}, f, ensure_ascii=False)
                    f.write('\n')

    def process_and_export(self, output_path, max_words=750):
        print("📘 Kitap verisi işleniyor...")
        chunks = self.split_into_chunks(max_words=max_words)
        self.export_jsonl(output_path, chunks)
        print(f"✅ {len(chunks)} adet parça {output_path} dosyasına yazıldı.")


class DeepSeekModelLoader:
    def __init__(self, model_name):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None

    def load(self):
        print("🧠 Tokenizer ve model yükleniyor...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")
        print("✅ Model ve tokenizer yüklendi.")
        return self.tokenizer, self.model
            
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")
            print("✅ Model GPU'ya taşındı.")
        else:
            print("❌ GPU bulunamadı, CPU kullanılacak.")

        print("✅ Model ve tokenizer yüklendi.")
        return self.tokenizer, self.model

class DeepSeekFineTuner:
    def __init__(self, tokenizer, model, data_path, output_dir, max_length=1024):
        self.tokenizer = tokenizer
        self.model = model
        self.data_path = data_path
        self.output_dir = output_dir
        self.max_length = max_length
        self.metric = evaluate.load("perplexity", module_type="metric")

    def load_dataset(self):
        return load_dataset("json", data_files={"train": self.data_path}, split="train")

    def tokenize_data(self, dataset):
        return dataset.map(
            lambda e: self.tokenizer(e["output"], truncation=True, max_length=self.max_length),
            batched=True,
            remove_columns=["output"]
        )

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        # Compute cross-entropy loss
        shift_logits = torch.tensor(logits[:, :-1, :])
        shift_labels = torch.tensor(labels[:, 1:])
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        perplexity = torch.exp(loss)
        return {"perplexity": perplexity.item()}

    def train(self, batch_size=2, grad_accum=4, epochs=3, lr=2e-5):
        dataset = self.load_dataset()
        tokenized_dataset = self.tokenize_data(dataset)

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            num_train_epochs=epochs,
            save_steps=500,
            save_total_limit=2,
            logging_steps=100,
            learning_rate=lr,
            fp16=torch.cuda.is_available(),
            evaluation_strategy="epoch",  # <- epoch sonunda ölçüm yapılır
            report_to="none"
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
            eval_dataset=tokenized_dataset.select(range(100)),  # sadece küçük bir parça eval için
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics
        )

        print("🎯 Eğitim başlatılıyor...")
        trainer.train()
        trainer.save_model(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        print(f"📦 Model ve tokenizer {self.output_dir} klasörüne kaydedildi.")


def main():
    # 🔹 Yolları belirle
    input_path = r'C:\Users\w\Desktop\Kodlama\VsCode\HelloWorld.py\kitap.json'
    output_path = r'C:\Users\w\Desktop\Kodlama\VsCode\HelloWorld.py\kitap_finetune_main.jsonl'
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    output_dir = "./deepseek_finetuned_law"

    # 🔹 Kitap verisini işle
    processor = DeepSeekBookProcessor(input_path)
    book_info = processor.explore_book()
    if isinstance(book_info, dict):
        print(f"📗 Kitap Adı: {book_info['kitap_adi']}")
        print(f"📄 Sayfa Sayısı: {book_info['sayfa_sayisi']}")
        print(f"📝 Toplam Kelime: {processor.count_words()}")
        print(f"📌 Cümle Sayısı: {processor.count_sentences()}")
        print(f"📑 Paragraf Sayısı: {processor.count_paragraphs()}")
    processor.process_and_export(output_path, max_words=750)

    # 🔹 Modeli yükle
    model_loader = DeepSeekModelLoader(model_name)
    tokenizer, model = model_loader.load()

    # 🔹 Fine-tuning başlat
    finetuner = DeepSeekFineTuner(
        tokenizer=tokenizer,
        model=model,
        data_path=output_path,
        output_dir=output_dir
    )
    finetuner.train(batch_size=2, grad_accum=4, epochs=3)

if __name__ == '__main__':
    main()
