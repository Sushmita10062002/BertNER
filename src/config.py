from transformers import AutoTokenizer

MAX_LEN = 128
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 16
EPOCHS = 10
TRAINING_FILE = "../inputs/ner_dataset.csv"
MODEL_NAME = "bert-base-uncased"
MODEL_PATH = "../outputs/model.bin"
TOKENIZER = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    do_lower_case = True
)
