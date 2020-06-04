import pandas as pd
from surprise import SVD, KNNBasic, KNNBaseline
from surprise import accuracy
from collections import defaultdict
from surprise.model_selection import train_test_split
from surprise.model_selection import GridSearchCV, KFold



class recsysBase:
    data        = ''
    trainset    = ''
    testset     = ''
    algorithm   = ''
    algo        = ''
    predictions = ''
    
    def __init__(self, data, algorithm='svd', algo_options={}, testset_percent=0):
        if not data:
            return
            
        self.data           = data
        self.algorithm      = algorithm

        ##
        if testset_percent == 0:
            self.trainset   = self.data.build_full_trainset()
            self.testset    = self.trainset.build_anti_testset()
        else:
            self.trainset, self.testset = train_test_split(self.data, test_size=testset_percent)

        if self.algorithm == 'svd':
            self.algo = SVD()
        elif self.algorithm == 'knn_basic':
            self.algo = KNNBasic()
            

        self.algo.fit(self.trainset)
        


    def exec(self):
        self.step1()
        self.step2()
        self.step3()

    def step1(self):
        pass

    def step2(self):
        pass

    def step3(self):
        pass

    def compute_rmse(self):
        if not self.predictions:
            self.test()
        
        accuracy.rmse(self.predictions)

    def load_from_file(self, file_path='predictions.csv'):
        self.predictions = pd.read_csv(filepath)

    def save_to_file(self, file_path='predictions.csv'):
        pd.DataFrame(algo.predictions).to_csv(file_path, index=False)

    def benchmark(self):
        cross_validate(self.algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
        
    def tune(self, opt_field='rmse', param_grid =
             {'n_epochs': [5, 10],
              'lr_all': [0.002, 0.005],
              'reg_all': [0.4, 0.6]
             }, SHOW_RESULT=False):

        if self.algorithm == 'svd':
            gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3)

        ## Start tuning
        gs.fit(self.data)

        ## Save to self.algo
        self.algo = gs.best_estimator[opt_field]
        self.algo.fit(self.trainset)

        if SHOW_RESULT:
            # best RMSE score
            print(gs.best_score['rmse'])

            # combination of parameters that gave the best RMSE score
            print(gs.best_params['rmse'])

        return self

    def test(self):
        self.predictions = self.algo.test(self.testset)
        self.compute_rmse()

    def get_top_n(self, target_uid=None, n=10, SHOW_RESULT=False):
        '''Return the top-N recommendation for each user from a set of predictions.

        Args:
            predictions(list of Prediction objects): The list of predictions, as
                returned by the test method of an algorithm.
            n(int): The number of recommendation to output for each user. Default
                is 10.

        Returns:
        A dict where keys are user (raw) ids and values are lists of tuples:
            [(raw item id, rating estimation), ...] of size n.
        '''
        
        if target_uid:
            target_uid = str(target_uid)

        # Check if testset is valid
        if not self.predictions:
            self.predictions = self.algo.test(self.testset)

        # First map the predictions to each user.
        top_n = defaultdict(list)
        for uid, iid, true_r, est, _ in self.predictions:
            top_n[uid].append((iid, est))

        # Then sort the predictions for each user and retrieve the k highest ones.
        for uid, user_ratings in top_n.items():
            if target_uid and target_uid != uid:
                continue
            
            user_ratings.sort(key=lambda x: x[1], reverse=True)

            if target_uid:
                top_n       = user_ratings[:n]
                break
            else:
                top_n[uid]  = user_ratings[:n]

        # Print the recommended items for each user
        if SHOW_RESULT:
            try:
                for uid, user_ratings in top_n.items():
                    print(uid, [iid for (iid, _) in user_ratings])
            except:
                print(top_n)

        
        return top_n




    def precision_recall_at_k(self, target_uid=1, threshold=3.5, k=10, num_of_testset=5, SHOW_RESULT=True):
        if target_uid:
            target_uid = str(target_uid)

        
        kf = KFold(n_splits=num_of_testset)

        final_precision = []
        final_recalls   = []

        for trainset, testset in kf.split(self.data):
            self.algo.fit(trainset)
            predictions = self.algo.test(testset)

            
            '''Return precision and recall at k metrics for each user.'''
            # First map the predictions to each user.
            user_est_true = defaultdict(list)
            for uid, _, true_r, est, _ in predictions:
                user_est_true[uid].append((est, true_r))

            precisions = dict()
            recalls = dict()
            for uid, user_ratings in user_est_true.items():
                # Sort user ratings by estimated value
                user_ratings.sort(key=lambda x: x[0], reverse=True)

                # Number of relevant items
                n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

                # Number of recommended items in top k
                n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

                # Number of relevant and recommended items in top k
                n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                                      for (est, true_r) in user_ratings[:k])

                # Precision@K: Proportion of recommended items that are relevant
                precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 1

                # Recall@K: Proportion of relevant items that are recommended
                recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 1


            if SHOW_RESULT:
                print('Relevant: ' + str(sum(prec for prec in precisions.values()) / len(precisions)))
                print('Recommended: ' + str(sum(rec for rec in recalls.values()) / len(recalls)))


            final_precision.append(precisions[uid])
            final_recalls.append(recalls[uid])

        return final_precision, final_recalls
