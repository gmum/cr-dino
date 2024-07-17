# Adapted from: https://github.com/ml-jku/cloome/blob/main/src/notebooks/retrieval.ipynb
# Runds retrieval based on input features and output csv file with results

import glob

from itertools import chain
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader

from tqdm import tqdm

# from sklearn.metrics import accuracy_score, top_k_accuracy_score

from data.img_mol_dataset import ImageMolDataset

# from huggingface_hub import hf_hub_download

def get_metrics(image_features, text_features):
    metrics = {}
    logits_per_image = image_features @ text_features.t()
    logits_per_text = logits_per_image.t()

    logits = {"image_to_text": logits_per_image.t(), "text_to_image": logits_per_text.t()}
    ground_truth = (
        torch.arange(len(text_features)).view(-1, 1).to(logits_per_image.device)
    )
    rankings = {}
    all_top_samples = {}
    all_preds = {}

    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        rankings[name] = ranking
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        all_preds[name] = preds
        top_samples = np.where(preds < 10)[0]
        all_top_samples[name] = top_samples
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)

    return rankings, all_top_samples, all_preds, metrics, logits

def read_features(dataset):
    labels = []
    img = []
    mol = []
    for batch in tqdm(DataLoader(dataset, num_workers=1, batch_size=64)):
        b0, b1, b2 = batch

        b1 = b1 / b1.norm(dim=-1, keepdim=True)
        b2 = b2 / b2.norm(dim=-1, keepdim=True)
        labels.append(b0)
        img.append(b1)
        mol.append(b2)

    return torch.cat(img), torch.cat(mol), list(chain.from_iterable(labels))

def run_retrieval(logits, type: str = 'img2text'):
    if type == 'img2text':
        key = "image_to_text"
    elif type == 'text2img':
        key = 'text_to_image'
    else:
        raise ValueError(f'Wrong type, expected [img2text, text2img], got {type}.')
    all_preds = []

    for i, logs in enumerate(logits[key]):
        choices = np.arange(len(mols))
        choices = np.delete(choices, i)
            
        logs = logs.cpu().numpy()
        
        positive = logs[i]
        negatives_ind = np.random.choice(choices, 99, replace=False)
        negatives = logs[negatives_ind]
        
        sampled_logs = np.hstack([positive, negatives])
        
        ground_truth = np.zeros(len(sampled_logs))
        ground_truth[0] = 1
        
        ranking = np.argsort(sampled_logs)
        ranking = np.flip(ranking)
        pred = np.where(ranking == 0)[0]
        all_preds.append(pred)


    all_preds = np.vstack(all_preds)
    # print(all_preds)

    r1 = np.mean(all_preds < 1) * 100
    r5 = np.mean(all_preds < 5) * 100
    r10 = np.mean(all_preds < 10) * 100
    print(r1, r5, r10)

    n1 = len(np.where(all_preds < 1)[0])
    n5 = len(np.where(all_preds < 5)[0])
    n10 = len(np.where(all_preds < 10)[0])
    print(n1, n5, n10)
    return {'r1': r1, 'r5':r5, 'r10': r10, 'n1': n1, 'n5': n5, 'n10': n10}
        

if __name__ == '__main__':
    results = []
    img_features_files = glob.glob('')
    for img_features in img_features_files:
        for split in range(5):
            print(f'Features: {img_features.split("/")[-1]}, split: {split}')
            print(' ----------------- Prepare data  -----------------')
            dataset = ImageMolDataset(
                img_path = img_features,
                mol_path = '',
                label_file = '',
                split=f''
            )
            imgs, mols, labels = read_features(dataset)
            print(imgs.shape, mols.shape, len(labels))
            print(' ----------------- Get metrics  -----------------')
            rankings, all_top_samples, all_preds, metrics, logits = get_metrics(imgs, mols)
            print(metrics)
            ground_truth = (
            torch.arange(len(mols)).view(-1, 1).to("cpu")
            )
            print(' ----------------- Image to text  -----------------')
            results_dict_img2text = run_retrieval(logits=logits, type='img2text')

            print(' ----------------- Text to image  -----------------')
            results_dict_text2img = run_retrieval(logits=logits, type='text2img')
            results.append([img_features.split('/')[-1], split, 'img2text', results_dict_img2text['r1'], results_dict_img2text['r5'], results_dict_img2text['r10']])
            results.append([img_features.split('/')[-1], split, 'text2img', results_dict_text2img['r1'], results_dict_text2img['r5'], results_dict_text2img['r10']])

    df = pd.DataFrame(data=results, columns=['features', 'split', 'retrieval', 'r1', 'r5', 'r10'])
    df.to_csv('')