"""
Finetune MolMCL on a custom dataset.

This script enables finetuning a pretrained MolMCL model on user-provided datasets.
"""

import os
import sys
import copy
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset, Subset
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch_geometric.transforms as T
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from sklearn.metrics import roc_auc_score, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from molmcl.finetune.model import GNNPredictor
from molmcl.splitters import scaffold_split
from molmcl.utils.data import (
    mol_to_graph_data_obj_basic,
    mol_to_graph_data_obj_rich,
    mol_to_graph_data_obj_super_rich,
)


class CustomMoleculeDataset(Dataset):
    """
    Dataset class for custom molecular data.
    """
    def __init__(self, smiles_list, labels, feat_type='super_rich'):
        self.feat_type = feat_type
        self.transform = T.AddRandomWalkPE(walk_length=20, attr_name='pe')

        feat_func = {
            'basic': mol_to_graph_data_obj_basic,
            'rich': mol_to_graph_data_obj_rich,
            'super_rich': mol_to_graph_data_obj_super_rich,
        }[feat_type]

        self.smiles = []
        self.labels = []
        self.mol_data = []

        for i, smi in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                data = feat_func(mol)
                self.smiles.append(smi)
                self.labels.append(labels[i])
                self.mol_data.append(self.transform(data))

        self.labels = np.array(self.labels)
        if len(self.labels.shape) == 1:
            self.labels = self.labels.reshape(-1, 1)
        self.num_task = self.labels.shape[1]

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        graph = self.mol_data[idx]
        graph.label = torch.Tensor(self.labels[idx])
        graph.smi = self.smiles[idx]
        return graph


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_optimizer(model, lr_params):
    pretrain_name, prompt_name, finetune_name = [], [], []
    for name, param in model.named_parameters():
        if 'gnn' in name or 'aggr' in name:
            pretrain_name.append(name)
        elif 'graph_pred_linear' in name:
            finetune_name.append(name)
        else:
            prompt_name.append(name)

    pretrain_params = list(
        map(lambda x: x[1], list(filter(lambda kv: kv[0] in pretrain_name, model.named_parameters()))))
    finetune_params = list(
        map(lambda x: x[1], list(filter(lambda kv: kv[0] in finetune_name, model.named_parameters()))))
    prompt_params = list(
        map(lambda x: x[1], list(filter(lambda kv: kv[0] in prompt_name, model.named_parameters()))))

    optimizer = torch.optim.Adam([
        {'params': finetune_params},
        {'params': pretrain_params, 'lr': lr_params['pretrain_lr']},
        {'params': prompt_params, 'lr': lr_params['prompt_lr']}
    ], lr=lr_params['finetune_lr'], weight_decay=lr_params['decay'])

    return optimizer


def train_epoch(model, train_loader, criterion, optimizer, scheduler, device, task_type, gradient_clip=5):
    model.train()
    loss_history = []

    for batch in train_loader:
        batch = batch.to(device)
        output = model(batch)
        predict = output['predict']
        label = batch.label.view(predict.shape)

        if task_type == 'classification':
            mask = label == 0
            loss = criterion(predict.double(), (label + 1) / 2) * (~mask)
            loss = loss.sum() / (~mask).sum()
        else:
            mask = torch.isnan(label)
            if mask.any():
                loss = criterion(predict, torch.nan_to_num(label, nan=0.0)) * (~mask)
                loss = loss.sum() / (~mask).sum()
            else:
                loss = criterion(predict, label).mean()

        optimizer.zero_grad()
        loss.backward()
        if gradient_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        loss_history.append(loss.item())

    return np.mean(loss_history)


def evaluate(model, loader, device, task_type):
    model.eval()
    y_true, y_scores = [], []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            predict = model(batch)['predict']
            y_true.append(batch.label.view(predict.shape))
            y_scores.append(predict)

    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim=0).cpu().numpy()

    if task_type == 'regression':
        valid_mask = ~np.isnan(y_true)
        if y_true.shape[1] == 1:
            y_t = y_true[valid_mask]
            y_s = y_scores[valid_mask]
            rmse = np.sqrt(mean_squared_error(y_t, y_s))
            r2 = r2_score(y_t, y_s) if len(y_t) > 1 else 0.0
            return {'rmse': rmse, 'r2': r2, 'metric': r2}
        else:
            rmse_list, r2_list = [], []
            for i in range(y_true.shape[1]):
                valid_i = valid_mask[:, i]
                if valid_i.sum() > 1:
                    y_t = y_true[valid_i, i]
                    y_s = y_scores[valid_i, i]
                    rmse_list.append(np.sqrt(mean_squared_error(y_t, y_s)))
                    r2_list.append(r2_score(y_t, y_s))
            avg_rmse = np.mean(rmse_list) if rmse_list else float('inf')
            avg_r2 = np.mean(r2_list) if r2_list else 0.0
            return {'rmse': avg_rmse, 'r2': avg_r2, 'metric': avg_r2}
    else:
        roc_list = []
        for i in range(y_true.shape[1]):
            if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == -1) > 0:
                is_valid = y_true[:, i] ** 2 > 0
                roc_list.append(roc_auc_score((y_true[is_valid, i] + 1) / 2, y_scores[is_valid, i]))
        auc = np.mean(roc_list) if roc_list else 0.5
        return {'auc': auc, 'metric': auc}


def main():
    parser = argparse.ArgumentParser(description='Finetune MolMCL on custom dataset')
    parser.add_argument('--input', '-i', required=True,
                        help='Input CSV file with SMILES and target columns')
    parser.add_argument('--output_dir', '-o', default='./results',
                        help='Output directory for checkpoints and results')
    parser.add_argument('--smiles_col', default='smiles',
                        help='Column name for SMILES (default: smiles)')
    parser.add_argument('--target_col', required=True, nargs='+',
                        help='Column name(s) for target values')
    parser.add_argument('--checkpoint', '-c', default='./checkpoint/zinc-gps_best.pt',
                        help='Path to pretrained checkpoint')
    parser.add_argument('--backbone', default='gps', choices=['gnn', 'gps'],
                        help='Model backbone (default: gps)')
    parser.add_argument('--feat_type', default='super_rich',
                        choices=['basic', 'rich', 'super_rich'],
                        help='Feature type (default: super_rich)')
    parser.add_argument('--task', default='classification',
                        choices=['classification', 'regression'],
                        help='Task type (default: classification)')
    parser.add_argument('--split', default='scaffold',
                        choices=['scaffold', 'random', 'predefined'],
                        help='Data split method (default: scaffold)')
    parser.add_argument('--test_file', default=None,
                        help='Separate test file (enables predefined split)')
    parser.add_argument('--split_col', default=None,
                        help='Column name indicating split (train/val/test) for predefined split')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs (default: 100)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate for prediction head (default: 0.001)')
    parser.add_argument('--pretrain_lr', type=float, default=0.0005,
                        help='Learning rate for pretrained layers (default: 0.0005)')
    parser.add_argument('--prompt_lr', type=float, default=0.0005,
                        help='Learning rate for prompt module (default: 0.0005)')
    parser.add_argument('--weight_decay', type=float, default=1e-6,
                        help='Weight decay (default: 1e-6)')
    parser.add_argument('--val_frac', type=float, default=0.1,
                        help='Validation fraction (default: 0.1)')
    parser.add_argument('--test_frac', type=float, default=0.1,
                        help='Test fraction (default: 0.1)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--device', default='cuda',
                        help='Device (default: cuda)')
    parser.add_argument('--no_prompt', action='store_true',
                        help='Disable prompt mechanism')
    parser.add_argument('--verbose', action='store_true',
                        help='Print training progress')
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    if args.test_file:
        args.split = 'predefined'

    if args.split == 'predefined' and args.test_file:
        train_df = pd.read_csv(args.input)
        test_df = pd.read_csv(args.test_file)

        if args.task == 'classification':
            train_labels = train_df[args.target_col].replace(0, -1).fillna(0).values
            test_labels = test_df[args.target_col].replace(0, -1).fillna(0).values
        else:
            train_labels = train_df[args.target_col].values.astype(np.float32)
            test_labels = test_df[args.target_col].values.astype(np.float32)

        train_dataset = CustomMoleculeDataset(
            train_df[args.smiles_col].tolist(), train_labels, feat_type=args.feat_type
        )
        test_dataset = CustomMoleculeDataset(
            test_df[args.smiles_col].tolist(), test_labels, feat_type=args.feat_type
        )

        n_train = len(train_dataset)
        val_size = int(n_train * args.val_frac)
        indices = list(range(n_train))
        random.shuffle(indices)
        val_idx = indices[:val_size]
        train_idx = indices[val_size:]
        train_dataset_split = Subset(train_dataset, train_idx)
        val_dataset = Subset(train_dataset, val_idx)

        n_missing = np.isnan(train_labels).sum() + np.isnan(test_labels).sum() if args.task == 'regression' else 0
        print(f'Loaded train: {len(train_df)}, test: {len(test_df)} molecules with {len(args.target_col)} task(s)')
        if n_missing > 0:
            print(f'Note: {n_missing} missing target values will be masked during training')
        print(f'Valid molecules - Train: {len(train_dataset)}, Test: {len(test_dataset)}')
        print(f'Train: {len(train_dataset_split)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}')

        train_dataset = train_dataset_split
        num_tasks = len(args.target_col)

    elif args.split == 'predefined' and args.split_col:
        df = pd.read_csv(args.input)
        smiles_list = df[args.smiles_col].tolist()

        if args.task == 'classification':
            labels = df[args.target_col].replace(0, -1).fillna(0).values
        else:
            labels = df[args.target_col].values.astype(np.float32)

        dataset = CustomMoleculeDataset(smiles_list, labels, feat_type=args.feat_type)

        split_values = df[args.split_col].values
        train_idx = [i for i, s in enumerate(split_values) if s == 'train']
        val_idx = [i for i, s in enumerate(split_values) if s == 'val']
        test_idx = [i for i, s in enumerate(split_values) if s == 'test']

        if not val_idx:
            n_train = len(train_idx)
            val_size = int(n_train * args.val_frac)
            random.shuffle(train_idx)
            val_idx = train_idx[:val_size]
            train_idx = train_idx[val_size:]

        train_dataset = Subset(dataset, train_idx)
        val_dataset = Subset(dataset, val_idx)
        test_dataset = Subset(dataset, test_idx)

        n_missing = np.isnan(labels).sum() if args.task == 'regression' else (labels == 0).sum()
        print(f'Loaded {len(smiles_list)} molecules with {len(args.target_col)} task(s)')
        if n_missing > 0:
            print(f'Note: {n_missing} missing target values will be masked during training')
        print(f'Valid molecules: {len(dataset)}')
        print(f'Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}')
        num_tasks = dataset.num_task

    else:
        df = pd.read_csv(args.input)
        smiles_list = df[args.smiles_col].tolist()

        if args.task == 'classification':
            labels = df[args.target_col].replace(0, -1).fillna(0).values
        else:
            labels = df[args.target_col].values.astype(np.float32)

        n_missing = np.isnan(labels).sum() if args.task == 'regression' else (labels == 0).sum()
        print(f'Loaded {len(smiles_list)} molecules with {len(args.target_col)} task(s)')
        if n_missing > 0:
            print(f'Note: {n_missing} missing target values will be masked during training')

        dataset = CustomMoleculeDataset(smiles_list, labels, feat_type=args.feat_type)
        print(f'Valid molecules: {len(dataset)}')

        if args.split == 'scaffold':
            train_idx, val_idx, test_idx = scaffold_split(
                dataset.smiles, frac_valid=args.val_frac, frac_test=args.test_frac, balanced=False
            )
        else:
            indices = list(range(len(dataset)))
            train_idx, temp_idx = train_test_split(
                indices, test_size=args.val_frac + args.test_frac, random_state=args.seed
            )
            val_idx, test_idx = train_test_split(
                temp_idx, test_size=args.test_frac / (args.val_frac + args.test_frac),
                random_state=args.seed
            )

        train_dataset = Subset(dataset, train_idx)
        val_dataset = Subset(dataset, val_idx)
        test_dataset = Subset(dataset, test_idx)

        print(f'Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}')
        num_tasks = dataset.num_task

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    if args.feat_type == 'basic':
        atom_feat_dim, bond_feat_dim = None, None
    elif args.feat_type == 'rich':
        atom_feat_dim, bond_feat_dim = 143, 14
    else:
        atom_feat_dim, bond_feat_dim = 170, 14

    model = GNNPredictor(
        num_layer=5,
        emb_dim=300,
        num_tasks=num_tasks,
        normalize=False,
        atom_feat_dim=atom_feat_dim,
        bond_feat_dim=bond_feat_dim,
        drop_ratio=0,
        attn_drop_ratio=0.3 if args.backbone == 'gps' else 0,
        temperature=0.5,
        use_prompt=not args.no_prompt,
        model_head=6,
        layer_norm_out=True,
        backbone=args.backbone,
    )

    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f'Loading pretrained checkpoint from {args.checkpoint}')
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        if 'wrapper' in checkpoint:
            model.load_state_dict(checkpoint['wrapper'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)

    model.to(device)
    model.freeze_aggr_module()

    lr_params = {
        'finetune_lr': args.lr,
        'pretrain_lr': args.pretrain_lr,
        'prompt_lr': args.prompt_lr,
        'decay': args.weight_decay,
    }
    optimizer = get_optimizer(model, lr_params)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0.0001)

    if args.task == 'regression':
        criterion = nn.MSELoss(reduction='none')
    else:
        criterion = nn.BCEWithLogitsLoss(reduction='none')

    best_metric = -float('inf')
    best_checkpoint = None
    results = []

    for epoch in tqdm(range(1, args.epochs + 1), desc='Training'):
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, None, device, args.task
        )
        scheduler.step()

        val_metrics = evaluate(model, val_loader, device, args.task)
        test_metrics = evaluate(model, test_loader, device, args.task)

        results.append({
            'epoch': epoch,
            'train_loss': train_loss,
            **{f'val_{k}': v for k, v in val_metrics.items()},
            **{f'test_{k}': v for k, v in test_metrics.items()},
        })

        if args.verbose:
            if args.task == 'classification':
                tqdm.write(f"[ep{epoch}] loss={train_loss:.4f} val_auc={val_metrics['auc']:.4f} test_auc={test_metrics['auc']:.4f}")
            else:
                tqdm.write(f"[ep{epoch}] loss={train_loss:.4f} val_r2={val_metrics.get('r2', 0):.4f} test_r2={test_metrics.get('r2', 0):.4f}")

        if val_metrics['metric'] > best_metric:
            best_metric = val_metrics['metric']
            best_checkpoint = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_checkpoint)
    final_test = evaluate(model, test_loader, device, args.task)

    print(f'\nFinal Test Results:')
    for k, v in final_test.items():
        if k != 'metric':
            print(f'  {k}: {v:.4f}')

    checkpoint_path = os.path.join(args.output_dir, 'best_model.pt')
    torch.save({
        'model_state_dict': best_checkpoint,
        'config': {
            'backbone': args.backbone,
            'feat_type': args.feat_type,
            'task': args.task,
            'num_tasks': num_tasks,
            'use_prompt': not args.no_prompt,
        }
    }, checkpoint_path)
    print(f'Best model saved to {checkpoint_path}')

    results_df = pd.DataFrame(results)
    results_path = os.path.join(args.output_dir, 'training_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f'Training results saved to {results_path}')


if __name__ == '__main__':
    main()

