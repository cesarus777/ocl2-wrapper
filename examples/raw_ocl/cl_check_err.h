//-----------------------------------------------------------------------------
//
// OpenCL errors handler header file
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

#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 210
#endif

#include <CL/cl.h>



void cl_handle_return_value(cl_int ret_value, const char *filename, int line);

#define CL_CHECK_RET(ret) cl_handle_return_value(ret, __FILE__, __LINE__);

