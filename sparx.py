import numpy as np
from scipy.sparse import csr_matrix
import time

# Step 3: Generate Sparse Matrix
dense_matrix = np.array([[1, 0, 0], [0, 0, 2], [0, 3, 0]])
sparse_matrix = csr_matrix(dense_matrix)

# Step 4: Perform Sparse Matrix Operations
vector = np.array([1, 2, 3])
result_sparse = sparse_matrix.dot(vector)

# Step 5: Optimize Computations
dense_matrix_2 = np.random.rand(1000, 1000)
sparse_matrix_2 = csr_matrix(dense_matrix_2)

# Measure execution time for dense matrix operation
start_time_dense = time.time()
result_dense = np.linalg.inv(dense_matrix_2)
end_time_dense = time.time()
print("Dense Matrix Operation Time:", end_time_dense - start_time_dense)

# Measure execution time for sparse matrix operation
start_time_sparse = time.time()
sparse_result = np.linalg.inv(sparse_matrix_2.toarray())
end_time_sparse = time.time()
print("Sparse Matrix Operation Time:", end_time_sparse - start_time_sparse)

# Step 6: Evaluate Performance
print("Sparse Matrix Result:\n", result_sparse)
print("Dense Matrix Result:\n", result_dense)



