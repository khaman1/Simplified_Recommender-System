from surprise import Dataset
from recsys import *


data = Dataset.load_builtin('ml-100k')


class MyTest(recsysBase):
    pass

## Get top-10 for recommendation
#print(MyTest(data).tune().get_top_n(n=10, target_uid=10, SHOW_RESULT=1))


##
#a,b = MyTest(data).precision_recall_at_k(target_uid=1, threshold=4.5, num_of_testset=2)

##
##MyTest(data).tune_and_test()

##
result = MyTest(data).get_k_neighbors(name='Forrest Gump (1994)', k=10)
