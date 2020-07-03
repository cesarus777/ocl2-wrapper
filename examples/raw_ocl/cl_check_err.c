//-----------------------------------------------------------------------------
//
// OpenCL errors handler
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

#include "cl_check_err.h"



void cl_handle_return_value(cl_int ret_value, const char *filename, int line)
{
  const char *problem = "unknown";

  switch (ret_value)
  {
  case CL_SUCCESS:
    return;
  case CL_BUILD_PROGRAM_FAILURE:
    problem = "build program failure";
    break;
  case CL_COMPILER_NOT_AVAILABLE:
    problem = "compiler not available";
    break;
  case CL_DEVICE_NOT_AVAILABLE:
    problem = "device not available";
    break;
  case CL_DEVICE_NOT_FOUND:
    problem = "device not found";
    break;
  case CL_INVALID_ARG_INDEX:
    problem = "invalid arg index";
    break;
  case CL_INVALID_ARG_SIZE:
    problem = "invalid arg size";
    break;
  case CL_INVALID_ARG_VALUE:
    problem = "invalid arg value";
    break;
  case CL_INVALID_BINARY:
    problem = "invalid binary";
    break;
  case CL_INVALID_BUILD_OPTIONS:
    problem = "invalid build options";
    break;
  case CL_INVALID_CONTEXT:
    problem = "invalid context";
    break;
  case CL_INVALID_DEVICE:
    problem = "invalid device";
    break;
  case CL_INVALID_DEVICE_TYPE:
    problem = "invalid device type";
    break;
  case CL_INVALID_DEVICE_QUEUE:
    problem = "invalid device queue";
    break;
  case CL_INVALID_KERNEL:
    problem = "invalid kernel";
    break;
  case CL_INVALID_KERNEL_NAME:
    problem = "invalid kernel name";
    break;
  case CL_INVALID_KERNEL_DEFINITION:
    problem = "invalid kernel definition";
    break;
  case CL_INVALID_MEM_OBJECT:
    problem = "invalid mem object";
    break;
  case CL_INVALID_OPERATION:
    problem = "invalid operation";
    break;
  case CL_INVALID_PLATFORM:
    problem = "invalid platform";
    break;
  case CL_INVALID_PROGRAM:
    problem = "invalid program";
    break;
  case CL_INVALID_PROGRAM_EXECUTABLE:
    problem = "invalid program executable";
    break;
  case CL_INVALID_PROPERTY:
    problem = "invalid property";
    break;
  case CL_INVALID_QUEUE_PROPERTIES:
    problem = "invalid queue properties";
    break;
  case CL_INVALID_SAMPLER:
    problem = "invalid sampler";
    break;
  case CL_INVALID_VALUE:
    problem = "invalid value";
    break;
  case CL_OUT_OF_HOST_MEMORY:
    problem = "out of host memory";
    break;
  case CL_OUT_OF_RESOURCES:
    problem = "out of resources";
    break;
  }

  fprintf(stderr, "Error: '%s' at %s:%d with return code %d\n",
                                        problem, filename, line, ret_value);
  exit(EXIT_FAILURE);
}

