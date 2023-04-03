import argparse
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping
from cnn import CNN, CNNTokenizer
from esm_model import ESM, ESMTokenizer
from data import DataModule, preprocessors
from csv import writer
from pathlib import Path
import os

RESULTS_DIR = 'results/'

def create_parser():
    parser = argparse.ArgumentParser(description="Train a flipper model")
    required = parser.add_argument_group('Required arguments')
    required.add_argument("dataset", choices = ["aav", "gb1"], help = "Choose the type of dataset", type=str)
    required.add_argument("split", help = "Choose the split within the dataset", type=str)
    required.add_argument("model", help = "Choose the model", choices = ["cnn", "esm1b", "esm1v"], type=str)
    required.add_argument("max_epochs", help = "Choose the max num of epochs", type=int)
    required.add_argument("batch_size", help = "Choose the batch size", type=int, default=256)
    required.add_argument("random_seed", help = "Choose the random seed", type=int, default=256)
    required.add_argument('-p', '--preprocess', action='store_true')
    return parser

# generally handle setting cpu/gpu
# have option to include preprocessing or not
# cnn fields - kernel_size, input_size, dropout
# esm fields - embedding_dim, pooling

if __name__ == "__main__":
    
    kernel_size = 5
    input_size = 43
    dropout = 0.0
    
    parser = create_parser()
    args = parser.parse_args()
    
    seed_everything(args.random_seed)

    if 'esm' in args.model:
        kernel_size, input_size, dropout = '', '', ''
    
    results_dir = Path(RESULTS_DIR)
    results_dir.mkdir(exist_ok=True)
    
    if args.model == 'cnn':
        model = CNN(kernel_size, input_size, dropout)
        tok = CNNTokenizer()
        early_stop = EarlyStopping(monitor='val_spearman', mode='max', patience=20)
    elif 'esm' in args.model:
        model = ESM(args.model, 1280)
        tok = ESMTokenizer(args.model)
        early_stop = EarlyStopping(monitor='val_loss', mode='min', patience=20)
        
    # max_epochs for cnn is 100, esm is 500
    dm = DataModule(args.dataset, '{}.csv'.format(args.split), args.batch_size, tok, args.preprocess)
    trainer = Trainer(callbacks=[early_stop], accelerator='gpu', devices=[0], max_epochs=args.max_epochs)
    # trainer = Trainer(callbacks=[EarlyStopping(monitor='val_spearman', mode='max', patience=20)], max_epochs=args.max_epochs) @ cpu
    trainer.fit(model, datamodule=dm)

    val_dict = trainer.validate(datamodule=dm)
    test_dict = trainer.test(datamodule=dm)

    log_num = sorted([int(s[8:]) for s in os.listdir('lightning_logs') if 'version_' in s])[-1]
    
    filename = results_dir / (args.dataset + '_results.csv')
    with open(filename, 'a', newline='') as f:
        w = writer(f)
        if not os.path.isfile(filename):
            w.writerow(['dataset', 'model', 'split', 'kernel size', 'input size', 'dropout', 'batch size', 'val spearman', 'val loss', 'test spearman', 'test loss', 'lightning log', 'preprocessed', 'random_seed'])
        w.writerow([args.dataset, args.model, args.split, kernel_size, input_size, dropout, args.batch_size, *val_dict[0].values(), *test_dict[0].values(), log_num, args.dataset in preprocessors and args.preprocess, args.random_seed])