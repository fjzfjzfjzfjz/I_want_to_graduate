import time

import numpy as np

c = time.time()
# for i in range(400000):
#     np.random.random()
# np.random.random(400000)
np.random.choice([1, 2, 3], 400000)
print()
print(time.time() - c)
