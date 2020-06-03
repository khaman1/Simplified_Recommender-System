from surprise import Dataset
from library.recsys import *


data = Dataset.load_builtin('ml-100k')


class MyTest(recsysBase):
    pass


algo = MyTest(data)

A = algo.get_top_n(n=10, target_uid=10)
