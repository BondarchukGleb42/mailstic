from transformers import AutoTokenizer
import torch
from sklearn.neighbors import KNeighborsClassifier
import pickle
import warnings
import os
from pathlib import Path
warnings.filterwarnings("ignore")
my_path = Path(__file__).parent


from .model import Model


def get_prediction(references_embeddings, reverences_labels, sample_text, tokenizer, model, max_length=512):
    sample_tokenized = tokenizer(sample_text, max_length=max_length, truncation=True, return_tensors='pt')
    sample_tokenized = {k: v.to(model.embedder.device) for k,v in sample_tokenized.items()}
    with torch.no_grad():
        sample_embedding = model(sample_tokenized)

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(references_embeddings.cpu(), reverences_labels)
    pred = knn.predict(sample_embedding.cpu())
    return pred[0]


def detect_defect_type(text, model_name):
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    model_id = "cointegrated/LaBSE-en-ru"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = Model(model_id).to(device)
    model.load_state_dict(torch.load(os.path.join(my_path, 'weights_few_shot.pt'),
                                     weights_only=True, map_location=device))
    model = model.eval()

    with open(os.path.join(my_path, f"user_models/{model_name}/all_embeddings_references.pkl"), "rb") as f:
        all_embeddings_references = pickle.load(f)
    
    with open(os.path.join(my_path, f"user_models/{model_name}/references_labels.pkl"), "rb") as f:
        references_labels = pickle.load(f)

    predict = get_prediction(all_embeddings_references, references_labels, text, tokenizer, model, 512)
    return predict

