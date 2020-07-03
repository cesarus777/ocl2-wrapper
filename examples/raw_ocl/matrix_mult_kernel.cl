__kernel void vec_add(__global int *A, __global int *B, __global int *C, int m)
{
  if(get_work_dim() != 2)
    return;

  size_t k = get_global_size(1);

  size_t i = get_global_id(0); // n index
  size_t j = get_global_id(1); // k index

  C[i * k + j] = 0;

  for(int l = 0; l < m; ++l)
  {
    C[i * k + j] += A[i * m + l] * B[l * k + j];
  }
}

