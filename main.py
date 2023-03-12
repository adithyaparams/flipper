import argparse
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping
from cnn import CNN, CNNTokenizer
from data import DataModule
from csv import writer
from pathlib import Path
import os

RESULTS_DIR = 'results/'

def create_parser():
    parser = argparse.ArgumentParser(description="train esm")
    parser.add_argument("dataset", type=str)
    parser.add_argument("split", type=str)
    parser.add_argument("model", choices = ["ridge", "cnn", "esm1b", "esm1v", "esm_rand"], type = str)
    parser.add_argument("max_epochs", type=int)
    return parser


if __name__ == "__main__":
    seed_everything(10)
    # random.seed(10)
    # torch.manual_seed(10)

    # dataset = 'aav'
    # model = 'cnn'
    # split = 'two_vs_many'
    kernel_size = 5
    input_size = 43
    dropout = 0.0
    batch_size = 64
    
    parser = create_parser()
    args = parser.parse_args()

    results_dir = Path(RESULTS_DIR)
    results_dir.mkdir(exist_ok=True)
    model = CNN(kernel_size, input_size, dropout)
    dm = DataModule(args.dataset, '{}.csv'.format(args.split), batch_size, CNNTokenizer())
    # max_epochs for cnn is 100, esm is 500
    trainer = Trainer(callbacks=[EarlyStopping(monitor='val_spearman', mode='max', patience=20)], accelerator='gpu', devices=[0], max_epochs=args.max_epochs)
    # trainer = Trainer(callbacks=[EarlyStopping(monitor='val_spearman', mode='max', patience=20)], max_epochs=args.max_epochs)
    # trainer = Trainer(callbacks=[EarlyStopping(monitor='val_loss', mode='min', patience=20)]) # ESM trainer
    trainer.fit(model, datamodule=dm)

    val_dict = trainer.validate(datamodule=dm)
    test_dict = trainer.test(datamodule=dm)

    with open(results_dir / (args.dataset+'_results.csv'), 'a', newline='') as f:  
        writer(f).writerow([args.dataset, args.model, args.split, kernel_size, input_size, dropout, batch_size, *val_dict[0].values(), *test_dict[0].values(), log_num])