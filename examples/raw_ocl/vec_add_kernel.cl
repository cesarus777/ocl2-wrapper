__kernel void vec_add(__global int *A, __global int *B,
                                       __global int *C, int size)
{
  size_t max_id = get_global_size(0);
  size_t i = get_global_id(0);
  
  while(i < size)
  {
    C[i] = A[i] + B[i];

    i += max_id;
  }
}

