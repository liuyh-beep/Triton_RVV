#ifdef C_KERNEL_ENABLE
#include "kernel/conv2d.h"
#endif

#ifdef TRITON_KERNEL_ENABLE
#include "conv2d_kernel_launcher.h"
#endif

#include "support/support.h"
#include <cassert>
#include <chrono>
#include <cstring>
#include <iostream>
#include <memory>
#include <random>
#include <stdio.h>
#include <stdlib.h>

using namespace std;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;

int main(int argc, char *argv[]) {
  int batch_size = 8;
  int channel_in = 64;
  int in_height = 56; 
  int in_width = 56;
  int channel_out = 64;
  int kernel_size = 7;
  int stride = 2;
  int padding = 3;
  int RUN_COUNT = 20;

  if (argc >= 2) {
    std::vector<int> Shape = splitStringToInts(argv[1]);

    if (Shape.size()) {
      assert(Shape.size() == 9 && "Invalid shape format: Batchs_sizexChannel_inxHeightxWidthxChannel_outxKernel_sizexStridexPaddingxDilationxGroupsxRUN_COUNT\n");
      batch_size = Shape.at(0);
      channel_in = Shape.at(1);
      in_height = Shape.at(2);
      in_width = Shape.at(3);
      channel_out = Shape.at(4);
      kernel_size = Shape.at(5);
      stride = Shape.at(6);
      padding = Shape.at(7);
      // dilation = 1 by default;
      // groups = 1 by default;
      RUN_COUNT = Shape.at(8);
    }
  }

  printf("Conv2D Data: Batch_size=%d, Channel_in=%d, Height=%d, Width=%d, Channel_out=%d, Kernel_size=%d, Stride=%d, Padding=%d, Dilation=1, Groups=1, RUN_COUNT=%d\n",
         batch_size, channel_in, in_height, in_width, channel_out,
         kernel_size, stride, padding, RUN_COUNT);

  float *arg0 = (float *)malloc(batch_size * channel_in * in_height * in_width * sizeof(float));

  float *arg1 = (float *)malloc(channel_out * channel_in * kernel_size * kernel_size * sizeof(float));

  float *arg2 = (float *)malloc(channel_out * sizeof(float));

  int out_height = (in_height + 2 * padding - (kernel_size - 1) - 1) / stride + 1;
  int out_width = (in_width + 2 * padding - (kernel_size - 1) - 1) / stride + 1;
  float *ref_out = (float *)malloc(batch_size * channel_out * out_height * out_width  * sizeof(float));

  float *real_out = (float *)malloc(batch_size * channel_out * out_height * out_width * sizeof(float));

  memset(real_out, 0, batch_size * channel_out * out_height * out_width * sizeof(float));

#ifdef CHECK_ACCURACY
    // convert to multi-D
    std::string file1 = getDB("conv2d", std::to_string(batch_size) + "x" + std::to_string(channel_in) + "x" + std::to_string(in_height) + "x" + std::to_string(in_width), 1);
    int M = batch_size * channel_in * in_height;
    int N = in_width;
    if (!readMatrix(file1.c_str(), arg0, M, N)) {
        printf("Failed to read first input matrix\n");
        return -1;
    }
    printf("Matrix 1 (%dx%dx%dx%d) loaded from %s\n", batch_size, channel_in, in_height, in_width, file1.c_str());

    std::string file2 = getDB("conv2d", std::to_string(channel_out) + "x" + std::to_string(channel_in) + "x" + std::to_string(kernel_size) + "x" + std::to_string(kernel_size), 2);
    M = channel_out * channel_in * kernel_size;
    N = kernel_size;
    if (!readMatrix(file2.c_str(), arg1, M, N)) {
        printf("Failed to read second weight matrix\n");
        return -1;
    }
    printf("Matrix 2 (%dx%dx%dx%d) loaded from %s\n", channel_out, channel_in, kernel_size, kernel_size, file2.c_str());

    std::string file3 = getDB("conv2d", std::to_string(1) + "x" + std::to_string(channel_out), 3);
    M = 1;
    N = channel_out;
    if (!readMatrix(file3.c_str(), arg2, M, N)) {
        printf("Failed to read reference output matrix\n");
        return -1;
    }
    printf("Matrix 3 (%dx%d) loaded from %s\n", 1, channel_out, file3.c_str());

    std::string file4 = getDB("conv2d", std::to_string(batch_size) + "x" + std::to_string(channel_out) + "x" + std::to_string(out_height) + "x" + std::to_string(out_width), 4);
    M = batch_size * channel_out * out_height;
    N = out_width;
    if (!readMatrix(file4.c_str(), ref_out, M, N)) {
        printf("Failed to read reference output matrix\n");
        return -1;
    }
    printf("Reference matrix (%dx%dx%dx%d) loaded from %s\n", batch_size, channel_out, out_height, out_width , file4.c_str());
#else
    // Will be used to obtain a seed for the random number engine
    std::random_device rd;
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
    std::normal_distribution<> norm_dis(0, 1);

    for(int i=0; i < batch_size * in_height * in_width; ++i) {
      for (int j = 0; j < channel_in; ++j) {
          arg0[i * channel_in + j] = norm_dis(gen);
      }
    }

    for (int i = 0; i < channel_in; ++i) {
      for (int j = 0; j < channel_out * kernel_size * kernel_size; ++j) {
          arg1[i * channel_out * kernel_size * kernel_size + j] = norm_dis(gen);
      }
    }

    for (int i = 0; i < channel_out; ++i) {
      arg2[i] = norm_dis(gen);
    }

#endif

  // triton kernel
#ifdef TRITON_KERNEL_ENABLE
  int num_thread = 1;
#ifdef UNCOMPLETE
  num_thread= -1;
#endif // UNCOMPLETE
  high_resolution_clock::time_point beginTime = high_resolution_clock::now();
  for (int i = 0; i < RUN_COUNT; i++) {
    conv2d_kernel_wrap(
      ceil(batch_size * out_height * out_width / conv2d_kernel_BLOCK_NI_HO_WO), 
        ceil(channel_out / conv2d_kernel_BLOCK_CO), 1,
           num_thread, conv2d_kernel, arg0, arg1, real_out, arg2, 
           batch_size,
           in_height, in_width, channel_out, 
           out_height, out_width, 
           channel_in * in_height * in_width, in_height * in_width, in_width,
           channel_in * kernel_size * kernel_size, kernel_size * kernel_size, kernel_size,
           channel_out * out_height * out_width, out_height * out_width, out_width,
           channel_in, kernel_size, kernel_size,
           stride,
           padding);
  }
  high_resolution_clock::time_point endTime = high_resolution_clock::now();
  milliseconds timeInterval =
      std::chrono::duration_cast<milliseconds>(endTime - beginTime);

  std::chrono::duration<double> triton_correlation_time_interval =
      endTime - beginTime;
  /// NOTE: Format running time to generate performance report easily
  PRINT_KERNEL_RUNNING_TIME(TRITON_KERNEL,
                            triton_correlation_time_interval.count())
  
#endif

// c kernel
#ifdef C_KERNEL_ENABLE

  // high_resolution_clock::time_point beginTime = high_resolution_clock::now();
  // for (int i = 0; i < RUN_COUNT; i++) {
  //   matmul(arg0, arg1, real_out, M, N, K);
  // }
  // high_resolution_clock::time_point endTime = high_resolution_clock::now();

  // milliseconds timeInterval =
  //     std::chrono::duration_cast<milliseconds>(endTime - beginTime);

  // std::chrono::duration<double> c_correlation_time_interval =
  //     endTime - beginTime;
  // /// NOTE: Format running time to generate performance report easily
  // PRINT_KERNEL_RUNNING_TIME(C_KERNEL, c_correlation_time_interval.count())
#endif

#ifdef CHECK_ACCURACY
  check_tensor(ref_out, real_out, M * N, "out", 1e-3);
#endif


#ifdef KEEP_TEST_DATA
  char filename[256];
  bool success = true;

  snprintf(filename, sizeof(filename), "matrix_%dx%dx%dx%d_1.txt", batch_size, in_height, in_width, channel_in);
  int MM = batch_size * in_height * in_width;
  int NN = channel_in;
  if (!writeMatrix(filename, arg0, MM, NN)) {
      printf("Failed to save input arg0\n");
      success = false;
  }
  snprintf(filename, sizeof(filename), "matrix_%dx%dx%dx%d_2.txt", channel_out, channel_in, kernel_size, kernel_size);
  MM = channel_out * channel_in * kernel_size;
  NN = kernel_size;
  if (!writeMatrix(filename, arg1, MM, NN)) {
      printf("Failed to save weight arg1\n");
      success = false;
  }
  snprintf(filename, sizeof(filename), "matrix_%dx%d_3.txt", 1, channel_out);
  MM = 1;
  NN = channel_out;
  if (!writeMatrix(filename, arg2, MM, NN)) {
      printf("Failed to save bias arg2\n");
      success = false;
  }

  snprintf(filename, sizeof(filename), "matrix_%dx%dx%dx%d_4.txt", batch_size, channel_out, out_height, out_width);
  MM = batch_size * channel_out * out_height;
  NN = out_width;
  if (!writeMatrix(filename, ref_out, MM, NN)) {
      printf("Failed to save output ref_out\n");
      success = false;
  }
  if(!success)
     printf("Warning: Some matrices were not saved successfully\n");
#endif

  free(arg0);
  free(arg1);
  free(arg2);
  free(ref_out);
  free(real_out);
  return 0;
}
