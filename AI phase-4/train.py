import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments# Load a pre-trained model
model_name = "gpt2-large" 
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)


# Read the conversation dataset from a text file
data_file = "/content/preprocessed_dataset.txt"

with open(data_file, "r", encoding="utf-8") as file:
    dataset = file.read().splitlines()

# Tokenize the dataset and prepare it for training
input_ids = []
attention_mask = []

for example in dataset:
    encoded = tokenizer.encode(example, return_tensors="pt", max_length=1024, truncation=True)
    input_ids.append(encoded)
    attention_mask.append(torch.ones_like(encoded))  # Assuming all tokens are relevant

# Create a custom PyTorch dataset
class CustomDataset(Dataset):
    def __init__(self, input_ids, attention_mask):
        self.input_ids = input_ids
        self.attention_mask = attention_mask

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx]
        }

custom_dataset = CustomDataset(input_ids, attention_mask)

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./gpt2",  # Change to your desired output directory
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    save_steps=1000,
    save_total_limit=2,
)

# Create data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# Initialize Trainer and start training
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=custom_dataset,
)

# Start training
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./gpt2")
