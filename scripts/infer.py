"""
Inference script for MolMCL.

Performs predictions on custom datasets using a finetuned MolMCL model.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T

from molmcl.finetune.model import GNNPredictor
from molmcl.utils.data import (
    mol_to_graph_data_obj_basic,
    mol_to_graph_data_obj_rich,
    mol_to_graph_data_obj_super_rich,
)


def load_model(checkpoint_path, config):
    """
    Load a finetuned MolMCL model from checkpoint.

    :param checkpoint_path: Path to the finetuned model checkpoint.
    :param config: Model configuration dictionary.
    :return: Loaded model.
    """
    if config['dataset']['feat_type'] == 'basic':
        atom_feat_dim, bond_feat_dim = None, None
    elif config['dataset']['feat_type'] == 'rich':
        atom_feat_dim, bond_feat_dim = 143, 14
    elif config['dataset']['feat_type'] == 'super_rich':
        atom_feat_dim, bond_feat_dim = 170, 14
    else:
        raise NotImplementedError('Unrecognized feature type.')

    model = GNNPredictor(
        num_layer=config['model']['num_layer'],
        emb_dim=config['model']['emb_dim'],
        num_tasks=config.get('num_tasks', 1),
        normalize=config['model']['normalize'],
        atom_feat_dim=atom_feat_dim,
        bond_feat_dim=bond_feat_dim,
        drop_ratio=0,
        attn_drop_ratio=0,
        temperature=config['model']['temperature'],
        use_prompt=config['model']['use_prompt'],
        model_head=config['model']['heads'],
        layer_norm_out=config['model']['layernorm'],
        backbone=config['model']['backbone'],
    )

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'wrapper' in checkpoint:
        model.load_state_dict(checkpoint['wrapper'], strict=False)
    else:
        model.load_state_dict(checkpoint)

    return model


def smiles_to_graphs(smiles_list, feat_type='super_rich'):
    """
    Convert SMILES strings to PyG graph objects.

    :param smiles_list: List of SMILES strings.
    :param feat_type: Feature type (basic, rich, super_rich).
    :return: Tuple of (valid_smiles, graph_list, invalid_indices).
    """
    feat_func = {
        'basic': mol_to_graph_data_obj_basic,
        'rich': mol_to_graph_data_obj_rich,
        'super_rich': mol_to_graph_data_obj_super_rich,
    }[feat_type]

    transform = T.AddRandomWalkPE(walk_length=20, attr_name='pe')

    valid_smiles = []
    graph_list = []
    invalid_indices = []

    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            data = feat_func(mol)
            data = transform(data)
            valid_smiles.append(smi)
            graph_list.append(data)
        else:
            invalid_indices.append(i)

    return valid_smiles, graph_list, invalid_indices


def predict(model, graphs, device, batch_size=32, task_type='classification'):
    """
    Run predictions on graph data.

    :param model: Loaded MolMCL model.
    :param graphs: List of PyG graph objects.
    :param device: Device to run on.
    :param batch_size: Batch size for inference.
    :param task_type: 'classification' or 'regression'.
    :return: Numpy array of predictions.
    """
    model.eval()
    model.to(device)

    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)

    all_predictions = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            output = model(batch)
            preds = output['predict']

            if task_type == 'classification':
                preds = torch.sigmoid(preds)

            all_predictions.append(preds.cpu().numpy())

    return np.concatenate(all_predictions, axis=0)


def main():
    parser = argparse.ArgumentParser(description='MolMCL Inference')
    parser.add_argument('--input', '-i', required=True,
                        help='Input file (CSV with SMILES column or TXT with one SMILES per line)')
    parser.add_argument('--output', '-o', required=True,
                        help='Output CSV file for predictions')
    parser.add_argument('--checkpoint', '-c', required=True,
                        help='Path to finetuned model checkpoint')
    parser.add_argument('--config', required=False,
                        help='Path to config YAML (optional, will use defaults if not provided)')
    parser.add_argument('--smiles_col', default='smiles',
                        help='Column name for SMILES in CSV input (default: smiles)')
    parser.add_argument('--backbone', default='gps', choices=['gnn', 'gps'],
                        help='Model backbone (default: gps)')
    parser.add_argument('--feat_type', default='super_rich',
                        choices=['basic', 'rich', 'super_rich'],
                        help='Feature type (default: super_rich)')
    parser.add_argument('--task', default='classification',
                        choices=['classification', 'regression'],
                        help='Task type (default: classification)')
    parser.add_argument('--num_tasks', type=int, default=1,
                        help='Number of prediction tasks (default: 1)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for inference (default: 32)')
    parser.add_argument('--device', default='cuda',
                        help='Device (default: cuda)')
    parser.add_argument('--no_prompt', action='store_true',
                        help='Disable prompt mechanism')
    args = parser.parse_args()

    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        config['num_tasks'] = args.num_tasks
    else:
        config = {
            'model': {
                'backbone': args.backbone,
                'num_layer': 5,
                'emb_dim': 300,
                'heads': 6,
                'layernorm': True,
                'dropout_ratio': 0,
                'attn_dropout_ratio': 0,
                'temperature': 0.5,
                'use_prompt': not args.no_prompt,
                'normalize': False,
            },
            'dataset': {
                'feat_type': args.feat_type,
                'task': args.task,
            },
            'num_tasks': args.num_tasks,
        }

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    input_ext = os.path.splitext(args.input)[1].lower()
    if input_ext == '.csv':
        df = pd.read_csv(args.input)
        smiles_list = df[args.smiles_col].tolist()
    elif input_ext == '.txt':
        with open(args.input, 'r') as f:
            smiles_list = [line.strip() for line in f if line.strip()]
        df = pd.DataFrame({'smiles': smiles_list})
    else:
        raise ValueError(f'Unsupported input format: {input_ext}')

    print(f'Loaded {len(smiles_list)} molecules from {args.input}')

    print('Converting SMILES to graphs...')
    valid_smiles, graphs, invalid_indices = smiles_to_graphs(
        smiles_list, feat_type=config['dataset']['feat_type']
    )
    print(f'Valid molecules: {len(valid_smiles)}, Invalid: {len(invalid_indices)}')

    if invalid_indices:
        print(f'Warning: {len(invalid_indices)} invalid SMILES at indices: {invalid_indices[:10]}...')

    print(f'Loading model from {args.checkpoint}...')
    model = load_model(args.checkpoint, config)
    print('Model loaded successfully')

    print('Running predictions...')
    predictions = predict(
        model, graphs, device,
        batch_size=args.batch_size,
        task_type=config['dataset']['task']
    )

    result_df = pd.DataFrame({'smiles': valid_smiles})
    if predictions.shape[1] == 1:
        result_df['prediction'] = predictions.flatten()
    else:
        for i in range(predictions.shape[1]):
            result_df[f'prediction_{i}'] = predictions[:, i]

    result_df.to_csv(args.output, index=False)
    print(f'Predictions saved to {args.output}')


if __name__ == '__main__':
    main()


