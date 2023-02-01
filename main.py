from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from cnn import CNN
from gb1 import DataModule, vocab

if __name__ == "__main__":
    model = CNN(len(vocab), 5, 1024, 0.0)
    gb1 = DataModule('gb1', 'two_vs_rest.csv', 256)
    trainer = Trainer(callbacks=[EarlyStopping(monitor='spearmanr', mode='max', patience=20)])
    trainer.fit(model, datamodule=gb1)
