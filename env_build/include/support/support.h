#include <algorithm>
#include <cmath>
#include <memory>
#include <optional>
#include <string>
#include <vector>
#include <iostream>

#define STRINGIFY(x) #x
#define CHECK_PTR(ptr, base, size) \
    if ((char*)(ptr) < (char*)(base) || (char*)(ptr) >= (char*)(base) + (size)) { \
        printf("ERROR: Pointer " STRINGIFY(ptr) " (%p) out of bounds [%p, %p)\n", ptr, base, (char*)(base) + (size)); \
        abort(); \
    }

unsigned int next_power_of_2(unsigned int n);

template <typename T = float>
bool check_tensor(T *a, T *b, int n, const char *label, float threshold = 1e-4) {
  bool ok = true;

  int j = 0;
  for (int i = 0; i < n; i++) {

    if (std::abs(a[i] - b[i]) > threshold) {
      // printf("Mismatch at %d: %f != %f\n", i, a[i], b[i]);
      ok = false;
      if (j++ < 32) {
        std::cout << i << " : " << a[i] << " vs " << b[i] << std::endl;
      }
      // break;
    }
  }
  std::string ACC = ok ? "OK" : "NOT OK";
  printf("%s %s\n", label, ACC.c_str());
  return ok;
}

std::vector<int> splitStringToInts(const std::string &str,
                                   char delimiter = 'x');

bool getBoolEnv(const std::string &env);

std::optional<int64_t> getIntEnv(const std::string &env);

std::optional<std::string> getStringEnv(const std::string &env);

// Data base
// std::string getDB(const std::string &Shape);
std::string getDB(const std::string &kernel_name, const std::string &Shape, int index);

std::unique_ptr<uint32_t[][3]> get_all_grids(uint32_t gridX, uint32_t gridY,
                                             uint32_t gridZ);

bool readMatrix(const char* filename, float* matrix, int& M, int& N);

bool readMatrix(const char* filename, int* matrix, int& M, int& N);

bool writeMatrix(const char* filename, const float* data, int rows, int cols);

#define PRINT_KERNEL_RUNNING_TIME(Kernel, Value)                               \
  std::cerr << "Running " << Kernel << " Time: " << Value << " s" << std::endl;

const std::string TRITON_KERNEL = "Triton Kernel";
const std::string C_KERNEL = "C Kernel";

/**
 * @brief Reads a 1D array of long integers (labels) from a text file.
 * Assumes one label per line.
 * @param filename Path to the input file.
 * @param data Pointer to the pre-allocated array to store labels.
 * @param M Number of labels to read (size of the array).
 * @return true if reading was successful, false otherwise.
 */
 bool readLabels(const char* filename, long* data, int M);

 /**
  * @brief Writes a 1D array of long integers (labels) to a text file.
  * Writes one label per line.
  * @param filename Path to the output file.
  * @param data Pointer to the array containing labels.
  * @param M Number of labels to write (size of the array).
  * @return true if writing was successful, false otherwise.
  */
 bool writeLabels(const char* filename, const long* data, int M);
 
 /**
  * @brief Reads a 1D array of floats (e.g., loss) from a text file.
  * Assumes one float value per line.
  * @param filename Path to the input file.
  * @param data Pointer to the pre-allocated array to store floats.
  * @param M Number of floats to read (size of the array).
  * @return true if reading was successful, false otherwise.
  */
 bool readLoss(const char* filename, float* data, int M);
 
 /**
  * @brief Writes a 1D array of floats (e.g., loss) to a text file.
  * Writes one float value per line.
  * @param filename Path to the output file.
  * @param data Pointer to the array containing floats.
  * @param M Number of floats to write (size of the array).
  * @return true if writing was successful, false otherwise.
  */
 bool writeLoss(const char* filename, const float* data, int M);