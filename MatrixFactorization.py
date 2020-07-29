import torch
##from https://github.com/yihong-chen/neural-collaborative-filtering/blob/master/src/gmf.py
import pytorch_lightning as pl
from pytorch_lightning.metrics import MSE
from torch.nn.functional import mse_loss

from torch.optim.lr_scheduler import ReduceLROnPlateau
from metrics import MetronAtK


class MatrixFactorization(pl.LightningModule):
    """
    Architecture modified from: https://github.com/yihong-chen/neural-collaborative-filtering
    Input: User tensor and Item tensor  //
    Output: predicted rating
    """

    def __init__(self, config):
        super(MatrixFactorization, self).__init__()
        self.config = config
        self._metron = MetronAtK(top_k= config['topk'])

        self.num_users = config['num_users']
        self.num_items = config['num_items']
        self.latent_dim = config['latent_dim']

        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)


    def forward(self, user_indices, item_indices):
        user_embedding = self.embedding_user(user_indices)
        item_embedding = self.embedding_item(item_indices)
        element_product = torch.bmm(user_embedding.unsqueeze(1), item_embedding.unsqueeze(1).permute(0, 2, 1)).squeeze()
        return element_product




    def use_optimizer(self, config):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=config['adam_lr'],
                                     weight_decay=config['l2_regularization'])
        return optimizer


    def training_step(self,batch,batch_idx):
        user,items,ratings,_ = batch[0],batch[1], batch[2], batch[3]
        ratings_pred = self(user,items)
        loss = mse_loss(ratings_pred.view(-1), ratings)
        return {'loss':loss}



    def validation_step(self,batch,batch_idx):
        test_users, test_items , test_true,exp_score = batch[0],batch[1], batch[2] , batch[3]
        test_pred = self(test_users, test_items)
        loss = mse_loss(test_pred.view(-1), test_true)

        return {'val_loss':loss,'test_users': test_users.detach(),'test_items':test_items.detach(),
                'test_pred':test_pred.detach(),'test_true':test_true.detach(),'exp_score':exp_score.detach() }

    def configure_optimizers(self):
        optimizer = self.use_optimizer(self.config)
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=15, factor=0.2, min_lr=1e-8)
        return [optimizer], [scheduler]

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        test_users = torch.cat([x['test_users'] for x in outputs])
        test_items = torch.cat([x['test_items'] for x in outputs])
        test_pred = torch.cat([x['test_pred'] for x in outputs])
        test_true = torch.cat([x['test_true'] for x in outputs])
        exp_score = torch.cat([x['exp_score'] for x in outputs])

        self._metron.subjects = [test_users.data.view(-1).tolist(),
                                 test_items.data.view(-1).tolist(),
                                 test_pred.data.view(-1).tolist(),
                                 test_true.data.view(-1).tolist(),
                                 exp_score.view(-1).tolist()]


        rmse, ndcg, map,mep = self._metron.cal_rmse(), self._metron.cal_ndcg(), self._metron.cal_map_at_k(),self._metron.calculate_MEP()
        print('[Evluating Epoch {}] RMSE = {:.4f}, NDCG = {:.4f}, MAP = {:.4f}, MEP = {:.4f}'.format(self.current_epoch, rmse, ndcg,map,mep))
        log = {'val_loss':avg_loss,'RMSE':rmse,'NDCG':ndcg,'MEP':mep}

        return {'log':log,'val_loss':ndcg,'rmse':rmse,'NDCG':ndcg,'MEP':mep}


    def init_weight(self):
        pass


