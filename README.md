# 1. Create conda environment
```
conda create -n molmcl python=3.10 -y
conda activate molmcl
```

# 2. Install PyTorch with CUDA
```
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
```

# 3. Install PyG and other dependencies
```
pip install torch_geometric -i https://pypi.org/simple/
pip install rdkit rogi matplotlib lmdb levenshtein cairosvg pandas pyyaml scipy scikit-learn -i https://pypi.org/simple/
```

# 4. Install MolMCL package
```
cd /home/ubuntu/MolMCL
pip install -e . --no-deps -i https://pypi.org/simple/
```

# 5. Download pretrained checkpoints
```
mkdir -p checkpoint
pip install gdown -i https://pypi.org/simple/
gdown --folder https://drive.google.com/drive/folders/1G_Yejbv8LCkV5guSf1WOJq2v3Nx55e58 --remaining-ok -O checkpoint/
mv checkpoint/molecule_multichannel_learning/checkpoint/*.pt checkpoint/

conda activate molmcl
export PYTHONPATH="/home/ubuntu/MolMCL:$PYTHONPATH"
```

# Then run training or inference scripts
```
python scripts/finetune_custom.py --help
python scripts/infer.py --help
```