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
    required.add_argument("dataset", type=str)
    required.add_argument("split", type=str)
    required.add_argument("model", choices = ["ridge", "cnn", "esm1b", "esm1v", "esm_rand"], type = str)
    required.add_argument("max_epochs", type=int)
    required.add_argument("batch_size", type=int, default=256)
    return parser


if __name__ == "__main__":
    seed_everything(10)
    
    kernel_size = 5
    input_size = 43
    dropout = 0.0
    
    parser = create_parser()
    args = parser.parse_args()

    if 'esm' in args.model:
        kernel_size, input_size, dropout = '', '', ''
    
    results_dir = Path(RESULTS_DIR)
    results_dir.mkdir(exist_ok=True)
    
    if args.model == 'cnn':
        model = CNN(kernel_size, input_size, dropout)
        tok = CNNTokenizer()
        early_stop = EarlyStopping(monitor='val_spearman', mode='max', patience=20)
    elif 'esm' in args.model:
        model = ESM('esm1v', 1280)
        tok = ESMTokenizer(args.model)
        early_stop = EarlyStopping(monitor='val_loss', mode='min', patience=20)
        
    # max_epochs for cnn is 100, esm is 500
    dm = DataModule(args.dataset, '{}.csv'.format(args.split), args.batch_size, tok)
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
            w.writerow(['dataset', 'model', 'split', 'kernel size', 'input size', 'dropout', 'batch size', 'val spearman', 'val loss', 'test spearman', 'test loss', 'lightning log', 'preprocessed'])
        w.writerow([args.dataset, args.model, args.split, kernel_size, input_size, dropout, args.batch_size, *val_dict[0].values(), *test_dict[0].values(), log_num, args.dataset in preprocessors])