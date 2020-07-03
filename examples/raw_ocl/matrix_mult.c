//-----------------------------------------------------------------------------
//
// Matrix multiplication in raw OpenCL
//
// A[n x m] * B[m x k] = C[n x k]
//
//-----------------------------------------------------------------------------
//
// Copyright © 2020 Yuly Tarasov. All rights reserved.
//
//-----------------------------------------------------------------------------
//
// This file is licensed after LGPL v3
// Look at: https://www.gnu.org/licenses/lgpl-3.0.en.html for details
//
//-----------------------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 210
#endif

#include <CL/cl.h>

#include "cl_check_err.h"

#ifndef STD_KERNEL_FILENAME
#define STD_KERNEL_FILENAME "matrix_mult_kernel.cl"
#endif

#ifndef CPU
# ifndef GPU
#  define GPU
# endif
#endif



enum { KERNEL_SOURCE_SIZE = 32768 };
enum { BUF_SIZE = 1024 };

enum { N = 2024, M = 2024, K = 2024 };


struct config_t
{
  cl_device_type type;
  int be_verbose;
  int with_timing;
  const char *kernel_filename;
};



struct config_t configurate(int argc, const char **argv);
cl_device_id detect_target_device_id(struct config_t config);



int main(int argc, const char **argv)
{
  printf("Running matrix_mult...\n");

  struct config_t config = configurate(argc, argv);
  cl_device_id target_device_id = detect_target_device_id(config);
  cl_int ret;

  if(config.be_verbose)
  {
    char device_name[BUF_SIZE];
    ret = clGetDeviceInfo(target_device_id, CL_DEVICE_NAME,
                                        sizeof(device_name), device_name, NULL);
    CL_CHECK_RET(ret);

    printf("Target device : %s\n", device_name);
  }

  cl_context context =
      clCreateContext(0, 1, &target_device_id, NULL, NULL, &ret);
  CL_CHECK_RET(ret);

#if CL_TARGET_OPENCL_VERSION < 200
  cl_command_queue command_queue =
                       clCreateCommandQueue(context, target_device_id, 0, &ret);
#else
  cl_command_queue command_queue =
      clCreateCommandQueueWithProperties(context, target_device_id, NULL, &ret);
#endif
  CL_CHECK_RET(ret);



  FILE *kernel_source = fopen(config.kernel_filename, "r");
  if(kernel_source == NULL)
  {
    fprintf(stderr, "Fatal error: can't open file '%s' with kernel\n",
                                                        config.kernel_filename);
    exit(EXIT_FAILURE);
  }

  char *kernel_source_str = (char *) malloc(KERNEL_SOURCE_SIZE);
  size_t kernel_source_size =
                 fread(kernel_source_str, 1, KERNEL_SOURCE_SIZE, kernel_source);
  fclose(kernel_source);

  cl_program program =
       clCreateProgramWithSource(context, 1, (const char **) &kernel_source_str,
                                    (const size_t *) &kernel_source_size, &ret);
  CL_CHECK_RET(ret);


  ret = clBuildProgram(program, 1, &target_device_id, NULL, NULL, NULL);
  CL_CHECK_RET(ret);


  cl_kernel kernel = clCreateKernel(program, "vec_add", &ret);
  CL_CHECK_RET(ret);



  int n = N, m = M, k = K;
  cl_int *A = (cl_int *) malloc(sizeof(cl_int) * n * m);
  cl_int *B = (cl_int *) malloc(sizeof(cl_int) * m * k);
  cl_int *C = (cl_int *) malloc(sizeof(cl_int) * n * k);
  cl_int *B_transposed = (cl_int *) malloc(sizeof(cl_int) * m * k);
  cl_int *C_CPU = (cl_int *) malloc(sizeof(cl_int) * n * k);

  for(int i = 0; i < n; ++i)
    for(int j = 0; j < m; ++j)
      A[i * m + j] = i + j;

  for(int i = 0; i < m; ++i)
  {
    for(int j = 0; j < k; ++j)
    {
      B[i * k + j] = (i * j) % (3 * (m + k));
      B_transposed[j * m + i] = (i * j) % (3 * (m + k));
    }
  }

  clock_t start, finish;
  start = clock();
  for(int i = 0; i < n; ++i)
  {
    for(int j = 0; j < k; ++j)
    {
      C_CPU[i * k + j] = 0;
      for(int l = 0; l < m; ++l)
      {
        C_CPU[i * k + j] += A[i * m + l] * B_transposed[j * m + l];
      }
    }
  }
  finish = clock();
  double cpu_time = (finish - start) / (double) CLOCKS_PER_SEC;
  free(B_transposed);
  if(config.with_timing)
  {
    printf("CPU calculating time: %gs\n", cpu_time);
  }



  cl_mem memobj_A = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                        sizeof(cl_int) * n * m, NULL, &ret);
  CL_CHECK_RET(ret);

  cl_mem memobj_B = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                        sizeof(cl_int) * m * k, NULL, &ret);
  CL_CHECK_RET(ret);

  cl_mem memobj_C = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                        sizeof(cl_int) * n * k, NULL, &ret);
  CL_CHECK_RET(ret);



  ret = clEnqueueWriteBuffer(command_queue, memobj_A, CL_TRUE, 0,
                                  sizeof(cl_int) * n * m, A, 0, NULL, NULL);
  CL_CHECK_RET(ret);

  ret = clEnqueueWriteBuffer(command_queue, memobj_B, CL_TRUE, 0,
                                  sizeof(cl_int) * m * k, B, 0, NULL, NULL);
  CL_CHECK_RET(ret);

  ret = clEnqueueWriteBuffer(command_queue, memobj_C, CL_TRUE, 0,
                                  sizeof(cl_int) * n * k, C, 0, NULL, NULL);
  CL_CHECK_RET(ret);



  ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *) &memobj_A);
  CL_CHECK_RET(ret);

  ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &memobj_B);
  CL_CHECK_RET(ret);

  ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *) &memobj_C);
  CL_CHECK_RET(ret);

  ret = clSetKernelArg(kernel, 3, sizeof(int), (void *) &m);
  CL_CHECK_RET(ret);



  const size_t global_work_size[2] = { n, k };

  if(config.be_verbose)
  {
    printf("Global work size : %lux%lu\n", global_work_size[0],
                                           global_work_size[1]);
  }

  start = clock();
  ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL,
                                         global_work_size, NULL, 0, NULL, NULL);
  CL_CHECK_RET(ret);


  ret = clEnqueueReadBuffer(command_queue, memobj_C, CL_TRUE, 0,
                                  sizeof(cl_int) * n * k , C, 0, NULL, NULL);
  CL_CHECK_RET(ret);
  finish = clock();
  double target_time = (finish - start) / (double) CLOCKS_PER_SEC;
  if(config.with_timing)
  {
    printf("Target device calculating time: %gs\n", target_time);
  }


  int errors = 0;

  if(config.be_verbose)
  {
    printf("Checking if calculations are correct...\n");
  }

  for(int i = 0; i < n * k; ++i)
  {
    if(C[i] != C_CPU[i])
    {
      printf("incorrect: C[%d:%d] == %d != %d\n", i / k, i % k, C[i], C_CPU[i]);
      ++errors;
    }
    else if(config.be_verbose)
    {
      printf("C[%d:%d] element is correct!\n", i / k, i % k);
    }

    if(errors > 20)
    {
      printf("Too many errors...\n");
      exit(EXIT_FAILURE);
    }
  }

  if(errors == 0)
  {
    printf("Added correctly!\n");
    exit(EXIT_SUCCESS);
  }
  else
  {
    printf("Error: %d errors in adding found!\n", errors);
    exit(EXIT_FAILURE);
  }
}



struct config_t configurate(int argc, const char **argv)
{
  if(argc < 1)
  {
    fprintf(stderr, "Fatal error: bad number of args in configuration\n");
    exit(EXIT_FAILURE);
  }

  struct config_t config;
#ifdef GPU
  config.type = CL_DEVICE_TYPE_GPU;
#else
  config.type = CL_DEVICE_TYPE_CPU;
#endif
  config.be_verbose = 0;
  config.with_timing = 0;
  config.kernel_filename = STD_KERNEL_FILENAME;

  if(argc == 1)
  {
    return config;
  }

  for(int i = 1; i < argc; ++i)
  {
    if((strcmp(argv[i], "-v") == 0) || (strcmp(argv[i], "--verbose") == 0))
    {
      config.be_verbose = 777;
    }
    else if((strcmp(argv[i], "-wt") == 0) || (strcmp(argv[i], "--with-timing") == 0))
    {
      config.with_timing = 777;
    }
    else if(strcmp(argv[i], "-k") == 0)
    {
      if((argc - (i + 1)) > 0)
      {
        config.kernel_filename = argv[++i];
        continue;
      }
      else
      {
        fprintf(stderr, "Error: missing filename after '-k'\n");
        exit(EXIT_FAILURE);
      }
    }
    else if(strncmp(argv[i], "--device=", 9) == 0)
    {
      const char *device = argv[i] + 9;
      if(strcmp(device, "GPU") == 0)
      {
        config.type = CL_DEVICE_TYPE_GPU;
      }
      else if(strcmp(device, "CPU") == 0)
      {
        config.type = CL_DEVICE_TYPE_CPU;
      }
      else
      {
        printf("Error: unrecognized device type: %s\n", device);
#ifdef GPU
        device = "GPU";
        config.type = CL_DEVICE_TYPE_GPU;
#else
        device = "CPU";
        config.type = CL_DEVICE_TYPE_CPU;
#endif
        printf("Setting default device type: %s\n", device);
      }
    }
    else
    {
      printf("Fatal error: unrecognized command line option '%s'\n", argv[i]);
      exit(EXIT_FAILURE);
    }
  }

  return config;
}



cl_device_id detect_target_device_id(struct config_t config)
{
  cl_device_id target_device_id;

  cl_int ret;
  cl_uint num_platforms;
  cl_uint num_devices;
  cl_platform_id *platform_ids;
  cl_device_id *device_ids;

  ret = clGetPlatformIDs(0, NULL, &num_platforms);
  CL_CHECK_RET(ret);

  if(config.be_verbose)
    printf("%d platform(s) found\n", num_platforms);

  platform_ids = (cl_platform_id *) malloc(num_platforms *
                                                        sizeof(cl_platform_id));
  ret = clGetPlatformIDs(num_platforms, platform_ids, NULL);
  CL_CHECK_RET(ret);

  for(size_t i = 0; i < num_platforms; ++i)
  {
    ret = clGetDeviceIDs(platform_ids[i], CL_DEVICE_TYPE_ALL,
                                                         0, NULL, &num_devices);
    CL_CHECK_RET(ret);

    device_ids = (cl_device_id *) malloc(num_devices * sizeof(cl_device_id));

    ret = clGetDeviceIDs(platform_ids[i], CL_DEVICE_TYPE_ALL,
                                                 num_devices, device_ids, NULL);
    CL_CHECK_RET(ret);

    for(size_t j = 0; j < num_devices; ++j)
    {
      cl_device_type type;

      if(config.be_verbose)
      {
        printf("platform # %lu, device # %lu\n", i, j);
        printf("device type(s) : ");
      }

      ret = clGetDeviceInfo(device_ids[j], CL_DEVICE_TYPE,
                                            sizeof(type), (void *) &type, NULL);
      CL_CHECK_RET(ret);

      for(size_t k = 0; k < 4; ++k)
      {
        cl_device_type one_type = type & (1u << k);
        if(config.be_verbose)
        {
          char *type_str;

          switch(one_type)
          {
          case CL_DEVICE_TYPE_DEFAULT:
            type_str = "default";
            break;
          case CL_DEVICE_TYPE_CPU:
            type_str = "CPU";
            break;
          case CL_DEVICE_TYPE_GPU:
            type_str = "GPU";
            break;
          case CL_DEVICE_TYPE_ACCELERATOR:
            type_str = "Accelerator";
            break;
          case CL_DEVICE_TYPE_CUSTOM:
            type_str = "custom";
            break;
          default:
            continue;
          }

          printf("%s", type_str);

          if((one_type >> k) > 1)
            printf(", ");
        }

        if(one_type == config.type)
          target_device_id = device_ids[j];
      }
    }

    if(config.be_verbose)
    {
      printf("\n");
    }

    free(device_ids);
  }

  free(platform_ids);

  return target_device_id;
}

