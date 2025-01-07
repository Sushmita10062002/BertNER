from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from model import EntityModel
import engine
import numpy as np
import dataset
import pandas as pd
import joblib
import config
import torch

def process_data(data_path):
    df = pd.read_csv(data_path, encoding = "latin-1")
    df = df.dropna(subset = ["Word"])
    df.loc[:, "Sentence #"] = df["Sentence #"].ffill()
    enc_pos = LabelEncoder()
    enc_tag = LabelEncoder()
    df.loc[:, "POS"] = enc_pos.fit_transform(df["POS"])
    df.loc[:, "Tag"] = enc_tag.fit_transform(df["Tag"])
    sentences = df.groupby("Sentence #")["Word"].apply(list).values
    pos = df.groupby("Sentence #")["POS"].apply(list).values
    tag = df.groupby("Sentence #")["Tag"].apply(list).values
    return sentences, pos, tag, enc_pos, enc_tag

if __name__ == "__main__":
    sentences, pos, tag, enc_pos, enc_tag = process_data(config.TRAINING_FILE)
    meta_data = {
        "enc_pos": enc_pos,
        "enc_tag": enc_tag
    }
    joblib.dump(meta_data, "/content/drive/MyDrive/projects/Named_Entity_Recognition/output/model.bin")

    num_pos = len(list(enc_pos.classes_))
    num_tag = len(list(enc_tag.classes_))

    (
        train_sentences,
        test_sentences,
        train_pos,
        test_pos,
        train_tag,
        test_tag
    ) = train_test_split(sentences, pos, tag, random_state = 42, test_size = 0.1)

    train_dataset = dataset.EntityDataset(
        texts = train_sentences, pos = train_pos, tags = train_tag
    )
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.TRAIN_BATCH_SIZE, num_workers=4
    )

    valid_dataset = dataset.EntityDataset(
        texts = test_sentences, pos = test_pos, tags = test_tag
    )
    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=config.VALID_BATCH_SIZE, num_workers=1
    )
    device = torch.device("cuda")
    model = EntityModel(num_tag = num_tag, num_pos = num_pos)
    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.001
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0
        }
    ]
    num_train_steps = int(len(train_sentences) / config.TRAIN_BATCH_SIZE * config.EPOCHS)
    optimizer = AdamW(optimizer_parameters, lr = 3e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )

    best_loss = np.inf
    for epoch in range(config.EPOCHS):
        train_loss = engine.train_fn(train_data_loader, model, optimizer, device, scheduler)
        test_loss = engine.eval_fn(valid_data_loader, model, device)
        print(f"Train loss = {train_loss} Valid loss = {test_loss}")
        if test_loss < best_loss:
            torch.save(model.state_dict(), config.MODEL_PATH)
            best_loss = test_loss
