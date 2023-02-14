from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from cnn import CNN, CNNTokenizer
from data import DataModule
from csv import writer
from pathlib import Path

RESULTS_DIR = 'results/'

if __name__ == "__main__":
    dataset = 'gb1'
    model = 'cnn'
    split = 'two_vs_rest'
    kernel_size = 5
    input_size = 1024
    dropout = 0.0
    batch_size = 256

    results_dir = Path(RESULTS_DIR)
    results_dir.mkdir(exist_ok=True)
    model = CNN(kernel_size, input_size, dropout)
    gb1 = DataModule('gb1', '{}.csv'.format(split), batch_size, CNNTokenizer())
    # max_epochs for cnn is 100, esm is 500
    trainer = Trainer(callbacks=[EarlyStopping(monitor='val_spearman', mode='max', patience=20)])
    # trainer = Trainer(callbacks=[EarlyStopping(monitor='val_loss', mode='min', patience=20)]) # ESM trainer
    trainer.fit(model, datamodule=gb1)

    val_dict = trainer.validate(datamodule=gb1)
    test_dict = trainer.test(datamodule=gb1)

    print(val_dict, test_dict)

    with open(results_dir / (dataset+'_results.csv'), 'a', newline='') as f:  
        writer(f).writerow([dataset, model, split, kernel_size, input_size, dropout, batch_size, *val_dict[0].values(), *test_dict[0].values()])