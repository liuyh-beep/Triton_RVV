#ifdef TRITON_KERNEL_ENABLE
#include "jacobi_kernel_launcher.h"
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
    int M = 1026;
    int N = 1026;
    int RUN_COUNT = 200;

    if (argc >= 2) {
        std::vector<int> Shape = splitStringToInts(argv[1]);

        if (Shape.size()) {
        assert(Shape.size() == 3 && "Invalid shape format: MxNxRUN_COUNT\n");
        M = Shape.at(0);
        N = Shape.at(1);
        RUN_COUNT = Shape.at(2);
        }
    }

    printf("Jacobi Data: M=%d, N=%d, RUN_COUNT=%d\n",
            M, N, RUN_COUNT);
        

    float *arg0 = (float *)malloc(M * N * sizeof(float));

    float *ref_out = (float *)malloc(M * N * sizeof(float));

    float *real_out = (float *)malloc(M * N * sizeof(float));

    memset(real_out, 0, M * N * sizeof(float));

#ifdef CHECK_ACCURACY
    // convert to multi-D
    std::string file1 = getDB("jacobi", std::to_string(M) + "x" + std::to_string(N), 1);
    if (!readMatrix(file1.c_str(), arg0, M, N)) {
        printf("Failed to read input matrix from %s\n", file1.c_str());
        return -1;
    }
    printf("Matrix 1 (%dx%d) loaded from %s\n", M, N, file1.c_str());

    std::string file2 = getDB("jacobi", std::to_string(M) + "x" + std::to_string(N), 2);
    if (!readMatrix(file2.c_str(), ref_out, M, N)) {
        printf("Failed to read reference output matrix from %s\n", file2.c_str());
        return -1;
    }
    printf("Matrix 2 (%dx%d) loaded from %s\n", M, N, file2.c_str());
#else
    // Will be used to obtain a seed for the random number engine
    std::random_device rd;
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
    std::normal_distribution<> norm_dis(0, 1);

    for(int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) {
          arg0[i * N + j] = norm_dis(gen);
      }
    }

#endif

  // triton kernel
#ifdef TRITON_KERNEL_ENABLE
  int num_thread = 1;
// #ifdef UNCOMPLETE
//   num_thread= -1;
// #endif // UNCOMPLETE
  high_resolution_clock::time_point beginTime = high_resolution_clock::now();
  for (int i = 0; i < RUN_COUNT; i++) {
    jacobi_kernel_wrap(
        ceil(1.0 * (M - 2) / jacobi_kernel_BLOCK_SIZE_M) * ceil(1.0 * (N - 2) / jacobi_kernel_BLOCK_SIZE_N),
        1, 1, num_thread,
        jacobi_kernel,
        arg0, real_out, M, N, N
    );
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

#ifdef CHECK_ACCURACY
  check_tensor(ref_out, real_out, M * N, "out", 1e-3);
#endif


#ifdef KEEP_TEST_DATA
  char filename[256];
  bool success = true;

  snprintf(filename, sizeof(filename), "matrix_%dx%d_1.txt", M, N);
  if (!writeMatrix(filename, arg0, M, N)) {
      printf("Failed to save input arg0\n");
      success = false;
  }
  snprintf(filename, sizeof(filename), "matrix_%dx%d_2.txt", M, N);
  if (!writeMatrix(filename, ref_out, M, N)) {
      printf("Failed to save output ref_out\n");
      success = false;
  }
  if(!success)
     printf("Warning: Some matrices were not saved successfully\n");
#endif

  free(arg0);
  free(ref_out);
  free(real_out);
  return 0;
}
