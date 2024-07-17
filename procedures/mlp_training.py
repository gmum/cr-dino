import os
import gin
from datetime import datetime
import numpy as np
from torch import optim, nn
import torch

from typing import Any, List
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from nn.masked_loss import masked_rocauc

@gin.configurable
class MLPTraining:
    def __init__(self,
                 model: nn.Module,
                 learning_rate: float,
                 weight_decay: float,
                 scheduler_step: int,
                 criterion: nn.Module,
                 tensorboard_dir: str,
                 model_name: str,
                 model_path: str,
                 optimizer: Any,
                 epochs: int,
                 save_epoch: int,
                 label_columns: List[str],
                 ):
        np.random.seed(101)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler_step = scheduler_step
        self.criterion = criterion().cuda()
        self.model_name = model_name + str(datetime.now()).replace(' ', '_')
        self.tb_writer = SummaryWriter(log_dir=os.path.join(tensorboard_dir, self.model_name))
        self.model_path = model_path + '_' + model_name + '_' + str(datetime.now()).replace(' ', '_')
        self.model = model
        self.optimizer = optimizer
        self.dataloaders = {}
        self.dataiters = {}
        self.epochs = epochs
        self.save_epoch = save_epoch
        self.softmax = nn.Softmax(dim=1)
        self.label_columns = label_columns
        
    def init_data(self, train_dataloader, test_dataloader):
        self.dataloaders = {'train': train_dataloader, 'test': test_dataloader}
        self.dataiters = {'train': iter(train_dataloader), 'test': iter(test_dataloader)}

    def get_next_batch(self, phase):
        try:
            return next(self.dataiters[phase])
        except StopIteration:
            self.dataiters[phase] = iter(self.dataloaders[phase])
            return next(self.dataiters[phase])
        
    def save_model(self, step):
        print('Saving model', str(os.path.join(self.model_path, f'{self.model_name}_{step}.pt')))
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        torch.save(self.model.state_dict(), os.path.join(self.model_path, f'{self.model_name}_{step}.pt'))

    def run(self,
            train_data_loader,
            test_data_loader):

        self.model = self.model.cuda()
        self.optimizer = self.optimizer(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.scheduler_step, gamma=0.1)

        for step in tqdm(range(self.epochs), total=self.epochs, desc='Epoch'):
            self.init_data(train_data_loader, test_data_loader)
            for phase in ['train', 'test']:
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()
                total_loss = 0
                labels = []
                preds = []
                for b, batch in enumerate(tqdm(self.dataiters[phase], total=len(self.dataiters[phase]), desc=phase)):
                    Xs, ys = batch
                    with torch.set_grad_enabled(phase == 'train'):
                        if phase == 'train':
                            self.optimizer.zero_grad()

                        Xs = Xs.cuda()
                        ys = ys.cuda()
                        outputs = self.model(Xs.float())
                        labels.append(ys.squeeze().detach().cpu().numpy())
                        preds.append(self.softmax(outputs).squeeze().detach().cpu().numpy())

                        loss = self.criterion(outputs.squeeze(), ys.squeeze().float())
                        total_loss += loss.item()
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                self.tb_writer.add_scalar(f'Loss/{phase}', total_loss / len(self.dataiters[phase]), step)

                labels = np.vstack(labels)
                preds = np.vstack(preds)
                roc_aucs = masked_rocauc(labels, preds)

                self.tb_writer.add_scalar(f'Avg ROC AUC/{phase}',
                                            np.mean(roc_aucs[roc_aucs>0]),
                                            step)
                self.tb_writer.add_scalar(f'ROC AUC > 0.7/{phase}',
                                            sum(roc_aucs > 0.7),
                                            step)
                self.tb_writer.add_scalar(f'ROC AUC > 0.8/{phase}',
                                            sum(roc_aucs > 0.8),
                                            step)
                self.tb_writer.add_scalar(f'ROC AUC > 0.9/{phase}',
                                            sum(roc_aucs > 0.9),
                                            step)
                # create dict with all roc auc values
                roc_auc_dict = dict(zip(self.label_columns, roc_aucs))
                for label, roc in roc_auc_dict.items():
                    self.tb_writer.add_scalar(f'ROC AUC {label}/{phase}',
                                               roc,
                                               step)
                    
                if (step + 1) % self.save_epoch == 0 and phase == 'train':
                    self.save_model(step)
                scheduler.step()
        