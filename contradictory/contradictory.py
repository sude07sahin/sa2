import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup
)
from datasets import load_dataset
import torchmetrics
from tqdm import tqdm
import matplotlib.pyplot as plt

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
def load_and_explore_data():
    # Load XNLI dataset
    dataset = load_dataset("xnli", "all_languages")
    
    # Explore dataset
    print("\nDataset structure:")
    print(dataset)
    
    print("\nExample from training set:")
    print(dataset["train"][0])
    
    print("\nLanguages available:")
    print(dataset["train"].features["language"].names)
    
    # Class distribution
    train_df = pd.DataFrame(dataset["train"])
    print("\nClass distribution in training set:")
    print(train_df["label"].value_counts(normalize=True))
    
    return dataset

dataset = load_and_explore_data()
class NLIDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        premise = self.data[idx]["premise"]
        hypothesis = self.data[idx]["hypothesis"]
        label = self.data[idx]["label"]
        
        encoding = self.tokenizer(
            premise,
            hypothesis,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "label": torch.tensor(label, dtype=torch.long)
        }

def prepare_data_loaders(dataset, tokenizer, batch_size=16):
    train_dataset = NLIDataset(dataset["train"], tokenizer)
    val_dataset = NLIDataset(dataset["validation"], tokenizer)
    test_dataset = NLIDataset(dataset["test"], tokenizer)
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )
    
    return train_loader, val_loader, test_loader
def initialize_model(num_labels=3):
    model_name = "xlm-roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    ).to(device)
    
    return model, tokenizer

model, tokenizer = initialize_model()
train_loader, val_loader, test_loader = prepare_data_loaders(dataset, tokenizer)
def setup_training(model, train_loader, epochs=3):
    optimizer = AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_loader) * epochs
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    loss_fn = torch.nn.CrossEntropyLoss().to(device)
    accuracy_metric = torchmetrics.Accuracy(task="multiclass", num_classes=3).to(device)
    f1_metric = torchmetrics.F1Score(task="multiclass", num_classes=3).to(device)
    
    return optimizer, scheduler, loss_fn, accuracy_metric, f1_metric

optimizer, scheduler, loss_fn, accuracy_metric, f1_metric = setup_training(model, train_loader)
def train_epoch(model, data_loader, optimizer, scheduler, loss_fn, accuracy_metric, f1_metric):
    model.train()
    losses = []
    correct_predictions = 0
    total_predictions = 0
    
    progress_bar = tqdm(data_loader, desc="Training", leave=False)
    
    for batch in progress_bar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        losses.append(loss.item())
        
        preds = torch.argmax(outputs.logits, dim=1)
        correct_predictions += torch.sum(preds == labels)
        total_predictions += len(labels)
        
        # Update metrics
        accuracy_metric.update(preds, labels)
        f1_metric.update(preds, labels)
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        progress_bar.set_postfix({"loss": loss.item()})
    
    accuracy = accuracy_metric.compute()
    f1 = f1_metric.compute()
    accuracy_metric.reset()
    f1_metric.reset()
    
    return np.mean(losses), accuracy.item(), f1.item()

def eval_epoch(model, data_loader, loss_fn, accuracy_metric, f1_metric):
    model.eval()
    losses = []
    correct_predictions = 0
    total_predictions = 0
    
    with torch.no_grad():
        progress_bar = tqdm(data_loader, desc="Validation", leave=False)
        
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            losses.append(loss.item())
            
            preds = torch.argmax(outputs.logits, dim=1)
            correct_predictions += torch.sum(preds == labels)
            total_predictions += len(labels)
            
            # Update metrics
            accuracy_metric.update(preds, labels)
            f1_metric.update(preds, labels)
            
            progress_bar.set_postfix({"loss": loss.item()})
    
    accuracy = accuracy_metric.compute()
    f1 = f1_metric.compute()
    accuracy_metric.reset()
    f1_metric.reset()
    
    return np.mean(losses), accuracy.item(), f1.item()

def train_model(model, train_loader, val_loader, optimizer, scheduler, loss_fn, accuracy_metric, f1_metric, epochs=3):
    history = {
        "train_loss": [],
        "train_acc": [],
        "train_f1": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": []
    }
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        
        train_loss, train_acc, train_f1 = train_epoch(
            model, train_loader, optimizer, scheduler, loss_fn, accuracy_metric, f1_metric
        )
        val_loss, val_acc, val_f1 = eval_epoch(
            model, val_loader, loss_fn, accuracy_metric, f1_metric
        )
        
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["train_f1"].append(train_f1)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")
    
    return history

history = train_model(
    model, train_loader, val_loader, 
    optimizer, scheduler, loss_fn, 
    accuracy_metric, f1_metric, epochs=3
)
def plot_training_history(history):
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history["train_acc"], label="Train Accuracy")
    plt.plot(history["val_acc"], label="Validation Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    
    plt.tight_layout()
    plt.show()

plot_training_history(history)

def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = torchmetrics.functional.accuracy(
        torch.tensor(all_preds), 
        torch.tensor(all_labels),
        task="multiclass", num_classes=3
    )
    
    f1 = torchmetrics.functional.f1_score(
        torch.tensor(all_preds), 
        torch.tensor(all_labels),
        task="multiclass", num_classes=3
    )
    
    print(f"\nTest Accuracy: {accuracy:.4f}")
    print(f"Test F1 Score: {f1:.4f}")
    
    return all_preds, all_labels

test_preds, test_labels = evaluate_model(model, test_loader)
def evaluate_by_language(model, dataset, tokenizer):
    languages = dataset["test"].features["language"].names
    results = []
    
    for lang in languages:
        lang_data = dataset["test"].filter(lambda x: x["language"] == languages.index(lang))
        lang_loader = DataLoader(
            NLIDataset(lang_data, tokenizer), 
            batch_size=16, 
            shuffle=False
        )
        
        _, labels = evaluate_model(model, lang_loader)
        preds = [p for p, l in zip(test_preds, test_labels) if l in [i for i, x in enumerate(lang_data["label"])]]
        labels = [l for l in test_labels if l in [i for i, x in enumerate(lang_data["label"])]]
        
        if len(preds) > 0:
            accuracy = torchmetrics.functional.accuracy(
                torch.tensor(preds), 
                torch.tensor(labels),
                task="multiclass", num_classes=3
            )
            
            f1 = torchmetrics.functional.f1_score(
                torch.tensor(preds), 
                torch.tensor(labels),
                task="multiclass", num_classes=3
            )
            
            results.append({
                "language": lang,
                "accuracy": accuracy.item(),
                "f1_score": f1.item(),
                "samples": len(preds)
            })
    
    results_df = pd.DataFrame(results)
    print("\nPerformance by Language:")
    print(results_df.sort_values("accuracy", ascending=False))
    
    return results_df

lang_results = evaluate_by_language(model, dataset, tokenizer)
def save_model(model, tokenizer, save_dir="./nli_model"):
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"Model saved to {save_dir}")

def load_model(save_dir="./nli_model"):
    tokenizer = AutoTokenizer.from_pretrained(save_dir)
    model = AutoModelForSequenceClassification.from_pretrained(save_dir).to(device)
    return model, tokenizer

save_model(model, tokenizer)
# model, tokenizer = load_model()
def predict_nli(premise, hypothesis, model, tokenizer):
    encoding = tokenizer(
        premise,
        hypothesis,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**encoding)
        pred = torch.argmax(outputs.logits, dim=1).item()
    
    label_map = {0: "entailment", 1: "neutral", 2: "contradiction"}
    return label_map[pred]

# Example usage
premise = "A man is eating pizza."
hypothesis = "Someone is having food."
prediction = predict_nli(premise, hypothesis, model, tokenizer)
print(f"\nPremise: {premise}")
print(f"Hypothesis: {hypothesis}")
print(f"Prediction: {prediction}")