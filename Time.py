import numpy as np
import time
from Tackle import repointcloud, KDEAN
from sklearn.preprocessing import MinMaxScaler
from gudhi.point_cloud.dtm import DistanceToMeasure
import random

# Creat 1000/2000/.../10000 random number list
random_list = [random.uniform(-1, 1) for _ in range(10000)]


SLICE = random_list

# only for time complexity testing
start_time = time.time()

pointcloud = repointcloud(len(SLICE), SLICE, 15, 8)
pointcloud_f = KDEAN(pointcloud, 5.2, 10)

# normalised data
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(pointcloud)

pointcloud_index = pointcloud_f[1]
outpoints2 = []
outpoints = []
for j in range(len(normalized_data)):
    for i2 in range(len(pointcloud_index)):
        if pointcloud_index[i2] == j:
            outpoints2.append(normalized_data[j])
            outpoints.append(pointcloud[j])

q = int(len(pointcloud_index) / 20) + 1
dtm = DistanceToMeasure(q)
# pointcloud = np.array(pointcloud)
outpoints2 = np.array(outpoints2)
outpoints = np.array(outpoints)
DTM_values = list(dtm.fit_transform(outpoints2))
DTM_values2 = list(dtm.fit_transform(outpoints))

# only for time complexity testing
end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")
