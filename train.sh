conda activate molmcl
export PYTHONPATH="/home/ubuntu/MolMCL:$PYTHONPATH"

python scripts/finetune_custom.py \
    --input ./data/finetune/expansionrx/train.csv \
    --test_file ./data/finetune/expansionrx/test.csv \
    --output_dir ./results/expansionrx \
    --smiles_col smiles \
    --target_col MPPB MBPB MGMB KSOL HLM_CLint MLM_CLint Caco2_Papp Caco2_Efflux LogD \
    --task regression \
    --epochs 100 \
    --verbose
