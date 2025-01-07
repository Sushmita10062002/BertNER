import numpy as np
import joblib
import torch
import config
import dataset
import engine
from model import EntityModel

from rich.console import Console
from rich.table import Table
from rich.text import Text

meta_data = joblib.load("../outputs/meta.bin")
enc_pos = meta_data["enc_pos"]
enc_tag = meta_data["enc_tag"]
num_pos = len(list(enc_pos.classes_))
num_tag = len(list(enc_tag.classes_))
device = torch.device("cuda")
model = EntityModel(num_tag = num_tag, num_pos = num_pos)
model.load_state_dict(torch.load(config.MODEL_PATH))
model.to(device)

console = Console()

def predict_ner(sentence):
    print("Sentence: ", sentence)
    tokenized_sentence = config.TOKENIZER.encode(sentence)
    sentence = sentence.split()
    test_dataset = dataset.EntityDataset(
        texts=[sentence],
        pos=[[0] * len(sentence)],
        tags=[[0] * len(sentence)]
    )

    token_ids = test_dataset[0]["ids"]
    tokens = config.TOKENIZER.convert_ids_to_tokens(token_ids)

    with torch.no_grad():
        data = test_dataset[0]
        for k, v in data.items():
            data[k] = v.to(device).unsqueeze(0)
        tag, pos, _ = model(**data)

        decoded_tags = enc_tag.inverse_transform(
            tag.argmax(2).cpu().numpy().reshape(-1)
        )[:len(tokenized_sentence)]
        decoded_pos = enc_pos.inverse_transform(
            pos.argmax(2).cpu().numpy().reshape(-1)
        )[:len(tokenized_sentence)]

        table = Table(title="NER Prediction", show_header=True, header_style="bold magenta")
        table.add_column("Token", style="cyan", justify="left")
        # table.add_column("POS", style="green", justify="left")
        table.add_column("NER Tag", style="yellow", justify="left")

        for token, tag, pos in zip(tokens[:len(tokenized_sentence)], decoded_tags, decoded_pos):
            table.add_row(token, tag)
        console.print(table)