from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from cnn import CNN
from gb1 import DataModule, vocab
from csv import writer

results_dir = 'results/'

if __name__ == "__main__":
    dataset = 'gb1'
    model = 'cnn'
    split = 'two_vs_rest'
    kernel_size = 5
    input_size = 1024
    dropout = 0.0
    batch_size = 256

    model = CNN(len(vocab), kernel_size, input_size, dropout)
    gb1 = DataModule('gb1', '{}.csv'.format(split), batch_size)
    trainer = Trainer(callbacks=[EarlyStopping(monitor='val_spearman', mode='max', patience=20)])
    trainer.fit(model, datamodule=gb1)

    val_dict = trainer.validate(datamodule=gb1)
    test_dict = trainer.test(datamodule=gb1)

    with open(results_dir / (dataset+'_results.csv'), 'a', newline='') as f:
        writer(f).writerow([dataset, model, split, kernel_size, input_size, dropout, batch_size, *val_dict.values(), *test_dict.values()])