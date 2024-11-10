import pandas as pd
from transformers import AutoTokenizer
import torch
from torch.utils.data import DataLoader
import pickle
import os
import warnings
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from transformers import AutoModel, AutoConfig

my_path = Path(__file__).parent

warnings.filterwarnings("ignore")

from .model import Model
from .torch_utils import EvalDataset, DataCollator


def add_new_class(new_class_name, texts):
    if os.path.exists(os.path.join(my_path, "data/references_labels.pkl")):
        with open(os.path.join(my_path, "data/references_labels.pkl"), "rb") as f:
            references_labels = pickle.load(f)
    else:
        references_labels = []

    if os.path.exists(os.path.join(my_path, "data/all_embeddings_references.pkl")):
        with open(
            os.path.join(my_path, "data/all_embeddings_references.pkl"), "rb"
        ) as f:
            all_embeddings_references = pickle.load(f)
    else:
        all_embeddings_references = []

    new_references = pd.DataFrame(
        {"full_text": texts, "target": [new_class_name] * len(texts)}
    )

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    model_id = "cointegrated/LaBSE-en-ru"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = Model(model_id).to(device)
    # model.load_state_dict(torch.load(os.path.join(my_path, 'weights_few_shot.pt'),
    # weights_only=True, map_location=device))

    model.load_state_dict(
        torch.load(
            os.path.join(my_path, "weights_few_shot.pt"),
            weights_only=True,
            map_location=device,
        )
    )
    model = model.eval()

    dataset_references = EvalDataset(new_references, target_column="target")
    data_collator = DataCollator(tokenizer=tokenizer, max_length=512)
    dataloader_references = DataLoader(
        dataset_references, batch_size=32, collate_fn=data_collator, shuffle=False
    )

    new_labels_references = []
    new_embeddings_references = [all_embeddings_references]

    for batch in tqdm(dataloader_references):
        batch["encodings"] = {k: v.to(device) for k, v in batch["encodings"].items()}

        with torch.no_grad():
            embeds = model(batch["encodings"])

        new_embeddings_references.append(embeds)
        new_labels_references.extend(batch["labels"])

    print(new_embeddings_references)
    embeddings_references = torch.cat(new_embeddings_references, dim=0)
    references_labels += new_labels_references

    fold_name = f"{new_class_name}"
    try:
        os.mkdir(os.path.join(my_path, f"user_models/{fold_name}/"))
    except FileExistsError:
        pass

    with open(
        os.path.join(my_path, f"user_models/{fold_name}/references_labels.pkl"), "wb"
    ) as f:
        pickle.dump(references_labels, f)

    with open(
        os.path.join(my_path, f"user_models/{fold_name}/all_embeddings_references.pkl"),
        "wb",
    ) as f:
        pickle.dump(embeddings_references, f)
