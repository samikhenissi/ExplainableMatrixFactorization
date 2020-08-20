import random
import pandas as pd
from scipy import spatial
from scipy.sparse import dok_matrix
from scipy.stats import stats
from sklearn.metrics import pairwise_distances
from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np

random.seed(42)

class UserItemRatingDataset(Dataset):
    """
    modified from: https://github.com/yihong-chen/neural-collaborative-filtering
    """

    """Wrapper, convert <user, item, rating, exp_score> Tensor into Pytorch Dataset"""

    def __init__(self, user_tensor, item_tensor, target_tensor,exp_score=None):
        """
        args:
            target_tensor: torch.Tensor, the corresponding rating for <user, item> pair
        """
        self.user_tensor = user_tensor
        self.item_tensor = item_tensor
        self.target_tensor = target_tensor
        self.exp_score = exp_score

    def __getitem__(self, index):
        if self.exp_score is not None:
            return self.user_tensor[index], self.item_tensor[index], self.target_tensor[index],self.exp_score[index]
        else:
            return self.user_tensor[index], self.item_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.user_tensor.size(0)




class SampleGenerator(object):

    def __init__(self, ratings,config):
        """
        Modified  from: https://github.com/yihong-chen/neural-collaborative-filtering
        Added batching for validation

        TODO: Add a random splitting
        """


        self.batch_size = config['batch_size']
        self.ratings = ratings

        self.n_users = len(ratings['userId'].unique())
        self.n_items = len(ratings['itemId'].unique())
        self.split_rate_test = config['test_ratio']
        self.user_pool = set(self.ratings['userId'].unique())
        self.item_pool = set(self.ratings['itemId'].unique())

        self.train_ratings, self.test_ratings = self._split(self.ratings)

        self.calc_exp(config['num_neighb'])

    def _split(self, ratings):
        """leave one out train/test split """
        ratings['rank_latest'] = ratings.groupby(['userId'])['timestamp'].rank(method='first', ascending=False)
        ratings['rank_max'] = ratings.groupby(['userId'])['rank_latest'].transform('max')
        ratings['number_test_rating'] = (ratings['rank_max'] * self.split_rate_test )
        ratings['number_test_rating'] = ratings['number_test_rating'].apply(lambda x: int(x))
        ratings['test'] = ratings['rank_latest'] <= ratings['number_test_rating']

        test = ratings[ratings['test'] == True]
        train = ratings[ratings['test'] == False ]

        print('train length is: ', len(train))
        print('test length is: ', len(test))

        assert train['userId'].nunique() == test['userId'].nunique()
        return train[['userId', 'itemId', 'rating']], test[['userId', 'itemId', 'rating']]


    def calc_exp(self, neighbor):  # calculate the explainability table
        rating_matrix = self.train_ratings.pivot(index='userId', columns='itemId', values='rating').fillna(0).values
        dist_items = pairwise_distances(rating_matrix, Y=None, metric='correlation')
        sorted_ind = np.argsort(dist_items, axis=1)
        def calculate_exp_score(i,j):
            sims = sorted_ind[i, 1:(neighbor + 1)]

            if j<rating_matrix.shape[1]:
                temp = [rating_matrix[y, j] for y in sims]
            else:
                temp = [0 for y in sims]
            num = sum(temp) / float(neighbor)
            return num
        self.train_ratings['exp_score'] = self.train_ratings[['userId','itemId']].apply(lambda x: calculate_exp_score(x[0],x[1]),axis=1)
        self.test_ratings['exp_score'] = self.test_ratings[['userId','itemId']].apply(lambda x: calculate_exp_score(x[0],x[1]),axis=1)


    def instance_a_train_loader_emf(self, batch_size):
        """instance train loader for one training epoch"""
        users, items, ratings, exp_score = self.train_ratings['userId'].tolist(), self.train_ratings['itemId'].tolist(), self.train_ratings['rating'].tolist(),  self.train_ratings['exp_score'].tolist()
        dataset = UserItemRatingDataset(user_tensor=torch.LongTensor(users),
                                        item_tensor=torch.LongTensor(items),
                                        target_tensor=torch.FloatTensor(ratings),
                                        exp_score=torch.FloatTensor(exp_score))
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)


    @property
    def instance_val_loader_emf(self):
        """create evaluate data"""
        test_users, test_items, test_output,  exp_score = self.test_ratings['userId'].tolist(), self.test_ratings['itemId'].tolist(),self.test_ratings['rating'].tolist(),self.test_ratings['exp_score'].tolist()

        dataset = UserItemRatingDataset(user_tensor=torch.LongTensor(test_users),
                                        item_tensor=torch.LongTensor(test_items),
                                        target_tensor=torch.FloatTensor(test_output),
                                        exp_score=torch.FloatTensor(exp_score))

        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False)


def preprocess_data(rating,config):
    """
    args:
        ratings: pd.DataFrame, which contains 4 columns = ['uid', 'mid', 'rating', 'timestamp']
        config: dict // name of the config dict.
    """
    assert 'uid' in rating.columns
    assert 'mid' in rating.columns
    assert 'rating' in rating.columns
    assert 'timestamp' in rating.columns

    user_id = rating[['uid']].drop_duplicates().reindex()
    user_id['userId'] = np.arange(len(user_id))
    rating = pd.merge(rating, user_id, on=['uid'], how='left')
    item_id = rating[['mid']].drop_duplicates()
    item_id['itemId'] = np.arange(len(item_id))
    rating = pd.merge(rating, item_id, on=['mid'], how='left')
    rating = rating[['userId', 'itemId', 'rating', 'timestamp']]
    print('Range of userId is [{}, {}]'.format(rating.userId.min(), rating.userId.max()))
    print('Range of itemId is [{}, {}]'.format(rating.itemId.min(), rating.itemId.max()))
    sample_generator = SampleGenerator(ratings=rating,config = config)

    print("created a generator object! ")
    return sample_generator

