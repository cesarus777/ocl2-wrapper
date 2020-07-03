//-----------------------------------------------------------------------------
//
// Listing OpenCL platforms
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

#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 210
#endif

#include <CL/cl.h>

#include "cl_check_err.h"

enum { BUF_SIZE = 2048 };

void print_platform_info(cl_platform_id pladtorm_id);
void print_device_info(cl_device_id device_id);

int main()
{
  cl_int ret;
  cl_uint num_platforms;
  cl_uint num_devices;
  cl_platform_id *platform_ids;
  cl_device_id *device_ids;

  ret = clGetPlatformIDs(0, NULL, &num_platforms);
  CL_CHECK_RET(ret);

  platform_ids = (cl_platform_id *) malloc(num_platforms * sizeof(cl_platform_id));
  ret = clGetPlatformIDs(num_platforms, platform_ids, NULL);
  CL_CHECK_RET(ret);

  for(size_t i = 0; i < num_platforms; ++i)
  {
    printf("Platform # %lu/%d\n", i + 1, num_platforms);
    print_platform_info(platform_ids[i]);
    printf("\n");

    ret = clGetDeviceIDs(platform_ids[i], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
    CL_CHECK_RET(ret);

    device_ids = (cl_device_id *) malloc(num_devices * sizeof(cl_device_id));

    ret = clGetDeviceIDs(platform_ids[i], CL_DEVICE_TYPE_ALL, num_devices, device_ids, NULL);
    CL_CHECK_RET(ret);

    for(size_t j = 0; j < num_devices; ++j)
    {
      printf("Device # %lu/%d\n", j + 1, num_devices);
      print_device_info(device_ids[j]);
      printf("\n");
    }

    free(device_ids);
  }

  free(platform_ids);
}

void print_platform_info(cl_platform_id platform_id)
{
  cl_int ret;
  char info[BUF_SIZE];

  ret = clGetPlatformInfo(platform_id, CL_PLATFORM_PROFILE, sizeof(info), info, NULL);
  CL_CHECK_RET(ret);
  printf("Platform profile: %s\n", info);

  ret = clGetPlatformInfo(platform_id, CL_PLATFORM_VERSION, sizeof(info), info, NULL);
  CL_CHECK_RET(ret);
  printf("Platform version: %s\n", info);

  ret = clGetPlatformInfo(platform_id, CL_PLATFORM_NAME, sizeof(info), info, NULL);
  CL_CHECK_RET(ret);
  printf("Platform name: %s\n", info);

  ret = clGetPlatformInfo(platform_id, CL_PLATFORM_VENDOR, sizeof(info), info, NULL);
  CL_CHECK_RET(ret);
  printf("Platform vendor: %s\n", info);

  ret = clGetPlatformInfo(platform_id, CL_PLATFORM_EXTENSIONS, sizeof(info), info, NULL);
  CL_CHECK_RET(ret);
  printf("Platform extensions: %s\n", info);
}

void print_device_info(cl_device_id device_id)
{
  cl_int ret;
  char info[BUF_SIZE];
  const char *info_ptr;

  ret = clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(info), info, NULL);
  CL_CHECK_RET(ret);
  printf("Device name: %s\n", info);

  size_t types_size;
  ret = clGetDeviceInfo(device_id, CL_DEVICE_TYPE, 0, NULL, &types_size);

  cl_device_type type;
  ret = clGetDeviceInfo(device_id, CL_DEVICE_TYPE, sizeof(type), &type, NULL);
  CL_CHECK_RET(ret);

  printf("Device type(s): ");

  for(int i = 0; i < 4; ++i)
  {
    cl_device_type one_type = type & (1u << i);
    switch(one_type)
    {
    case CL_DEVICE_TYPE_DEFAULT:
      info_ptr = "default";
      break;
    case CL_DEVICE_TYPE_CPU:
      info_ptr = "CPU";
      break;
    case CL_DEVICE_TYPE_GPU:
      info_ptr = "GPU";
      break;
    case CL_DEVICE_TYPE_ACCELERATOR:
      info_ptr = "Accelerator";
      break;
    case CL_DEVICE_TYPE_CUSTOM:
      info_ptr = "custom";
      break;
    default:
      continue;
    }

    printf("%s", info_ptr);

    if((one_type >> i) > 1)
      printf(", ");
  }


  ret = clGetDeviceInfo(device_id, CL_DEVICE_VENDOR, sizeof(info), info, NULL);
  CL_CHECK_RET(ret);
  printf("Device vendor: %s\n", info);

  ret = clGetDeviceInfo(device_id, CL_DEVICE_VERSION, sizeof(info), info, NULL);
  CL_CHECK_RET(ret);
  printf("Device version: %s\n", info);

  ret = clGetDeviceInfo(device_id, CL_DEVICE_PROFILE, sizeof(info), info, NULL);
  CL_CHECK_RET(ret);
  printf("Device profile: %s\n", info);

  cl_bool is_available;
  ret = clGetDeviceInfo(device_id, CL_DEVICE_AVAILABLE, sizeof(is_available),
                                                        &is_available, NULL);
  CL_CHECK_RET(ret);
  info_ptr = (is_available == CL_TRUE) ? "is available" : "is not available";
  printf("Device %s\n", info_ptr);
}

