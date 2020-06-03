from surprise import Dataset
from library.recsys import *


data = Dataset.load_builtin('ml-100k')


class MyTest(recsysBase):
    pass


algo = MyTest(data).tune(SHOW_RESULT=1).get_top_n(n=10, target_uid=10)
