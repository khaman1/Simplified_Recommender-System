from surprise import SVD, KNNBasic
from surprise import accuracy
from surprise.model_selection import train_test_split
from collections import defaultdict


class recsysBase:
    data        = ''
    trainset    = ''
    testset     = ''
    
    algo        = ''
    predictions = ''
    
    def __init__(self, data, algorithm='svd', testset_percent=.25):
        self.data   = data

        ##
        if testset_percent == 0:
            self.trainset   = self.data.build_full_trainset()
            self.testset    = self.trainset.build_anti_testset()
        else:
            self.trainset, self.testset = train_test_split(self.data, test_size=testset_percent)

        if algorithm == 'svd':
            self.algo = SVD()
        elif algorithm == 'knn_basic':
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

    def test(self):
        self.predictions = self.algo.test(self.testset)
        self.compute_rmse()

    def get_top_n(self, n=10, SHOW_RESULT=0):
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

        ## Check if testset is valid
        if not self.predictions:
            self.predictions = self.algo.test(self.testset)

        # First map the predictions to each user.
        top_n = defaultdict(list)
        for uid, iid, true_r, est, _ in self.predictions:
            top_n[uid].append((iid, est))

        # Then sort the predictions for each user and retrieve the k highest ones.
        for uid, user_ratings in top_n.items():
            user_ratings.sort(key=lambda x: x[1], reverse=True)
            top_n[uid] = user_ratings[:n]

        # Print the recommended items for each user
        if SHOW_RESULT:
            for uid, user_ratings in top_n.items():
                print(uid, [iid for (iid, _) in user_ratings])

        return top_n
