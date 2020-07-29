
from MatrixFactorization import *


class ExpMatrixFactorization(MatrixFactorization):
    def __init__(self, config):
        super(ExpMatrixFactorization, self).__init__(config)

        self.lamda = config['lamda']
    def training_step(self,batch,batch_idx):
        user,items,ratings,exp_score = batch[0],batch[1], batch[2],batch[3]
        ratings_pred = self(user,items)
        loss = mse_loss(ratings_pred.view(-1), ratings)
        P = self.embedding_user(user)
        Q = self.embedding_item(items)
        exp_reg = torch.norm(P-Q) * exp_score
        loss += exp_reg.mean() * self.lamda
        return {'loss':loss}


    def init_weight(self):
        pass



