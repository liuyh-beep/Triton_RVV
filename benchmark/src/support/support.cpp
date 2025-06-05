#include "support/support.h"
#include <cassert>
#include <sstream>

unsigned int next_power_of_2(unsigned int n) {
  if (n == 0) {
    return 1;
  }
  n--;
  n |= n >> 1;
  n |= n >> 2;
  n |= n >> 4;
  n |= n >> 8;
  n |= n >> 16;
  return n + 1;
}

std::vector<int> splitStringToInts(const std::string &str, char delimiter) {
  std::vector<int> result;
  std::stringstream ss(str);
  std::string temp;

  while (std::getline(ss, temp, delimiter)) {
    result.push_back(std::stoi(temp));
  }

  return result;
}

bool getBoolEnv(const std::string &env) {
  const char *s = std::getenv(env.c_str());
  std::string str(s ? s : "");
  std::transform(str.begin(), str.end(), str.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return str == "on" || str == "true" || str == "1";
}

std::optional<int64_t> getIntEnv(const std::string &env) {
  const char *cstr = std::getenv(env.c_str());
  if (!cstr)
    return std::nullopt;

  char *endptr;
  long int result = std::strtol(cstr, &endptr, 10);
  if (endptr == cstr)
    assert(false && "invalid integer");
  return result;
}

std::optional<std::string> getStringEnv(const std::string &env) {
  const char *cstr = std::getenv(env.c_str());
  if (!cstr)
    return std::nullopt;
  return std::string(cstr);
}

// std::string getDB(const std::string &Shape) {
//   std::string DB;
//   if (auto V = getStringEnv("DB_FILE"))
//     DB = V.value();
//   assert(DB.size());
//   DB += "_" + Shape + ".bin";
//   return DB;
// }

std::string getDB(const std::string &kernel_name, const std::string &Shape, int index) {
  std::string DB;
  // ../test_data/matrix_{shape}_{index}.txt

  DB = kernel_name + "/run/test_data/matrix_" + Shape + "_" + std::to_string(index) + ".txt";
  // "../test_data/matrix" + "_" + Shape + "_" + std::to_string(index) + ".txt";
  return DB;
}

bool readMatrix(const char* filename, float* matrix, int& M, int& N) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        printf("Cannot open file %s\n", filename);
        return false;
    }

    if (fscanf(file, "%d %d", &M, &N) != 2) {
        printf("Error reading matrix dimensions\n");
        fclose(file);
        return false;
    }

    for (int i = 0; i < M * N; i++) {
        if (fscanf(file, "%f", &matrix[i]) != 1) {
            printf("Error reading matrix data\n");
            fclose(file);
            return false;
        }
    }

    fclose(file);
    return true;
}

bool writeMatrix(const char* filename, const float* data, int rows, int cols) {
    FILE* fp = fopen(filename, "w");
    if (!fp) {
        printf("Failed to open file %s for writing\n", filename);
        return false;
    }

    fprintf(fp, "%d %d\n", rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            fprintf(fp, "%.6f ", data[i * cols + j]);
        }
        fprintf(fp, "\n");
    }

    fclose(fp);
    return true;
}

std::unique_ptr<uint32_t[][3]> get_all_grids(uint32_t gridX, uint32_t gridY,
                                             uint32_t gridZ) {
  std::unique_ptr<uint32_t[][3]> grids(new uint32_t[gridX * gridY * gridZ][3]);
  // TODO: which order would be more effective for cache locality?
  for (uint32_t z = 0; z < gridZ; ++z) {
    for (uint32_t y = 0; y < gridY; ++y) {
      for (uint32_t x = 0; x < gridX; ++x) {
        grids[z * gridY * gridX + y * gridX + x][0] = x;
        grids[z * gridY * gridX + y * gridX + x][1] = y;
        grids[z * gridY * gridX + y * gridX + x][2] = z;
      }
    }
  }
  return grids;
}
