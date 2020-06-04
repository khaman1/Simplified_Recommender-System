from surprise import Dataset
from library.recsys import *


data = Dataset.load_builtin('ml-100k')


class MyTest(recsysBase):
    pass


#print(MyTest(data).tune().get_top_n(n=10, target_uid=10, SHOW_RESULT=1))

a,b = MyTest(data).precision_recall_at_k(target_uid=1)
