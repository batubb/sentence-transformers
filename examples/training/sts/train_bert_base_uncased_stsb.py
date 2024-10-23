import argparse
import math

from scipy.stats import spearmanr

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from transformers.optimization import get_scheduler, SchedulerType


class SentencePairDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def pad_data(self, data):
        sent1 = [x["sentence1"] for x in data]
        sent2 = [x["sentence2"] for x in data]
        labels = [x["score"] for x in data]

        encoding1 = self.tokenizer(
            sent1, return_tensors="pt", padding=True, truncation=True
        )
        encoding2 = self.tokenizer(
            sent2, return_tensors="pt", padding=True, truncation=True
        )

        token_ids = torch.LongTensor(encoding1["input_ids"])
        attention_mask = torch.LongTensor(encoding1["attention_mask"])
        token_type_ids = torch.LongTensor(encoding1["token_type_ids"])

        token_ids2 = torch.LongTensor(encoding2["input_ids"])
        attention_mask2 = torch.LongTensor(encoding2["attention_mask"])
        token_type_ids2 = torch.LongTensor(encoding2["token_type_ids"])
        labels = torch.DoubleTensor(labels)

        return (
            token_ids,
            token_type_ids,
            attention_mask,
            token_ids2,
            token_type_ids2,
            attention_mask2,
            labels,
        )

    def collate_fn(self, all_data):
        (
            token_ids,
            token_type_ids,
            attention_mask,
            token_ids2,
            token_type_ids2,
            attention_mask2,
            labels,
        ) = self.pad_data(all_data)

        batched_data = {
            "token_ids_1": token_ids,
            "token_type_ids_1": token_type_ids,
            "attention_mask_1": attention_mask,
            "token_ids_2": token_ids2,
            "token_type_ids_2": token_type_ids2,
            "attention_mask_2": attention_mask2,
            "labels": labels,
        }

        return batched_data


class SBert(nn.Module):

    def __init__(self):
        super(SBert, self).__init__()
        self.originalModel = AutoModel.from_pretrained("google-bert/bert-base-uncased")

    def forward(self, input_ids, attention_mask):
        # This is N, T, D
        out = self.originalModel(input_ids, attention_mask)["last_hidden_state"]
        # pool, i.e. take the mean of the out
        attention_mask_expanded = attention_mask.unsqueeze(-1).expand(out.size())
        masked_embeddings = out * attention_mask_expanded
        # Compute the sum of the token embeddings, ignoring pad tokens
        sum_embeddings = masked_embeddings.sum(dim=1)
        # Count the number of non-pad tokens (sum of attention mask along sequence length)
        non_pad_tokens = attention_mask.sum(dim=1, keepdim=True)
        # Compute the average of non-pad token embeddings
        sentence_embeddings = sum_embeddings / non_pad_tokens

        return sentence_embeddings


class SimpleBert(nn.Module):
    def __init__(self, mode):
        super(SimpleBert, self).__init__()
        self.originalModel = AutoModel.from_pretrained("google-bert/bert-base-uncased")
        assert mode == "CLS" or mode == "AVG_IGNORE_PADS" or mode == "AVG"
        self.mode = mode

    def forward(self, input_ids, attention_mask):
        if self.mode == "CLS":
            return self.originalModel(input_ids, attention_mask)["pooler_output"]
        elif self.mode == "AVG_IGNORE_PADS":
            out = self.originalModel(input_ids, attention_mask)["last_hidden_state"]
            # pool, i.e. take the mean of the out
            attention_mask_expanded = attention_mask.unsqueeze(-1).expand(out.size())
            masked_embeddings = out * attention_mask_expanded
            # Compute the sum of the token embeddings, ignoring pad tokens
            sum_embeddings = masked_embeddings.sum(dim=1)
            # Count the number of non-pad tokens (sum of attention mask along sequence length)
            non_pad_tokens = attention_mask.sum(dim=1, keepdim=True)
            # Compute the average of non-pad token embeddings
            sentence_embeddings = sum_embeddings / non_pad_tokens
            return sentence_embeddings
        else:
            out = self.originalModel(input_ids, attention_mask)["last_hidden_state"]
            return out.mean(dim=1)


def evaluate(model, sts_dev_data_loader, device):
    model.eval()
    with torch.no_grad():
        all_cosine_similarities = []
        allLabels = []
        for batch in sts_dev_data_loader:

            b_ids1, b_mask1, b_ids2, b_mask2, b_labels = (
                batch["token_ids_1"],
                batch["attention_mask_1"],
                batch["token_ids_2"],
                batch["attention_mask_2"],
                batch["labels"],
            )
            b_ids1 = b_ids1.to(device)
            b_mask1 = b_mask1.to(device)
            b_ids2 = b_ids2.to(device)
            b_mask2 = b_mask2.to(device)
            b_labels = b_labels.to(device)
            # forward pass
            # out1 is of shape: B, D
            out1 = model(b_ids1, b_mask1)
            out2 = model(b_ids2, b_mask2)
            # cosine_similarities is of shape: B,
            cosine_similarities = F.cosine_similarity(out1, out2)
            # numpy version:
            # cosine_similarities = 1 - (
            #     paired_cosine_distances(out1.cpu().numpy(), out2.cpu().numpy())
            # )
            all_cosine_similarities.extend(cosine_similarities.cpu().numpy())
            allLabels.extend(b_labels.cpu().numpy())
        return spearmanr(allLabels, all_cosine_similarities)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--use_gpu", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda") if args.use_gpu else torch.device("cpu")

    # 1. prepare the training data
    sts_train_data = load_dataset("sentence-transformers/stsb", split="train")
    sts_train_dataset = SentencePairDataset(sts_train_data)
    # if you run on local, use a Subset
    # sts_train_dataset_subset = Subset(sts_train_dataset, range(40))
    BATCH_SIZE = 16
    sts_train_data_loader = DataLoader(
        sts_train_dataset,
        shuffle=False,
        batch_size=BATCH_SIZE,
        collate_fn=sts_train_dataset.collate_fn,
        num_workers=4 if args.use_gpu else 0,  # make this 4 when GPU is available,
    )

    sts_dev_data = load_dataset("sentence-transformers/stsb", split="validation")
    sts_dev_dataset = SentencePairDataset(sts_dev_data)
    # if you run on local, use a Subset
    # sts_dev_dataset_subset = Subset(sts_dev_dataset, range(40))
    sts_dev_data_loader = DataLoader(
        sts_dev_dataset,
        shuffle=True,
        batch_size=BATCH_SIZE,
        collate_fn=sts_dev_dataset.collate_fn,
        num_workers=4 if args.use_gpu else 0,  # make this 4 when GPU is available,
    )
    # 2. prepare the model
    model = SBert()
    model = model.to(device)
    num_epochs = 4

    # This is 0.0 because that's what HuggingFace default implementation does
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.0)
    num_training_steps = len(sts_train_data_loader) * num_epochs
    num_warmup_steps = num_training_steps * 0.1
    scheduler = get_scheduler(
        SchedulerType.LINEAR, optimizer, num_warmup_steps, num_training_steps
    )
    # 3. run training
    iteration_num = 1
    for _ in range(num_epochs):
        model.train()
        for batch in sts_train_data_loader:
            b_ids1, b_mask1, b_ids2, b_mask2, b_labels = (
                batch["token_ids_1"],
                batch["attention_mask_1"],
                batch["token_ids_2"],
                batch["attention_mask_2"],
                batch["labels"],
            )
            b_ids1 = b_ids1.to(device)
            b_mask1 = b_mask1.to(device)
            b_ids2 = b_ids2.to(device)
            b_mask2 = b_mask2.to(device)
            b_labels = b_labels.to(device)

            # forward pass
            out1 = model(b_ids1, b_mask1)
            out2 = model(b_ids2, b_mask2)
            cosine_similarity = F.cosine_similarity(out1, out2)
            loss = F.mse_loss(cosine_similarity.double(), b_labels)

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            iteration_num += 1
            if iteration_num % 100 == 0:
                print(
                    f"loss: {loss.item()} learning_rate: {scheduler.get_last_lr()[0]:.10f} {iteration_num=}"
                )
        spearman, _ = evaluate(model, sts_dev_data_loader, device)
        # TODO: this ends up being 4.002 and 3.002. I think we have a one off bug here but it's not that important
        epoch_in_ref_impl = iteration_num / len(sts_train_data_loader)
        print(f"{spearman=} {iteration_num=} {epoch_in_ref_impl=}")

    # Run the 'basic' bert-base-uncased model to see how that would've performed without any additional trainig we've done in SBert
    # To do this, we need to use bert-base-uncased to get the embedding(represantation) of a sentence. Let's look at how we did this in the CS224N assignment
    # If you look at the code in `/Users/batuhan.balci/Documents/CS224N Spring 24/CS224N-Spring2024-DFP-Student-Handout/classifier.py`, you will see that BertSentimentClassifier.forward
    # simply uses self.bert(...)['pooler_output']. So we can do the same
    basicModelCLS = SimpleBert("CLS")
    basicModelCLS = basicModelCLS.to(device)
    spearman_basic_cls, _ = evaluate(basicModelCLS, sts_dev_data_loader, device)
    print(
        f"spearman score with the only the untrained bert CLS token: {spearman_basic_cls=}"
    )

    basicModelAVGIgnorePads = SimpleBert("AVG_IGNORE_PADS")
    basicModelAVGIgnorePads = basicModelAVGIgnorePads.to(device)
    spearman_basic_avg_ignore_pads, _ = evaluate(
        basicModelAVGIgnorePads, sts_dev_data_loader, device
    )
    print(
        f"spearman score with the only the untrained bert avg of tokens ignore pads: {spearman_basic_avg_ignore_pads=}"
    )

    basicModelAVG = SimpleBert("AVG")
    basicModelAVG = basicModelAVG.to(device)
    spearman_basic_avg, _ = evaluate(basicModelAVG, sts_dev_data_loader, device)
    print(
        f"spearman score with the only the untrained bert avg of tokens: {spearman_basic_avg=}"
    )


if __name__ == "__main__":
    main()

# Run this and compare the result with python training_stsbenchmark.py bert-base-uncased
