#include "common.h"
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string.h>
#include <unistd.h>
#include <vector>
#include <vortex.h>

#define RT_CHECK(_expr)                                      \
  do {                                                       \
    int _ret = _expr;                                        \
    if (0 == _ret)                                           \
      break;                                                 \
    printf("Error: '%s' returned %d!\n", #_expr, (int)_ret); \
    cleanup();                                               \
    exit(-1);                                                \
  } while (false)

const char *kernel_file = "kernel.vxbin";
uint32_t matrix_size = 0;

vx_device_h device = nullptr;
vx_buffer_h A_buffer = nullptr;
vx_buffer_h B_buffer = nullptr;
vx_buffer_h C_buffer = nullptr;
vx_buffer_h kernel_buffer = nullptr;
vx_buffer_h args_buffer = nullptr;

std::vector<uint8_t> staging_buf;
kernel_arg_t *kernel_arg = {};

static void show_usage() {
  std::cout << "Vortex Test." << std::endl;
  std::cout << "Usage: [-k: kernel] [-n words] [-h: help]" << std::endl;
}

static void parse_args(int argc, char **argv, uint32_t &data_size) {
  int c;
  while ((c = getopt(argc, argv, "n:k:d:h?")) != -1) {
    switch (c) {
    case 'n':
      matrix_size = atoi(optarg);
      break;
    case 'k':
      kernel_file = optarg;
      break;
    case 'd':
      data_size = atoi(optarg);
      break;
    case 'h':
    case '?': {
      show_usage();
      exit(0);
    } break;
    default:
      show_usage();
      exit(-1);
    }
  }
}

void cleanup() {
  if (device) {
    vx_mem_free(A_buffer);
    vx_mem_free(B_buffer);
    vx_mem_free(C_buffer);
    vx_mem_free(kernel_buffer);
    vx_mem_free(args_buffer);
    vx_dev_close(device);
  }
}

template <typename TYPE>
class mainVars {

public:
  mainVars(uint32_t buf_size, uint32_t data_size, uint32_t matrix_size) : buf_size(buf_size),                                                                       
                                                                          matrix_size(matrix_size) {
    src_A.resize(matrix_size * matrix_size);
    src_B.resize(matrix_size * matrix_size);
    refs.resize(matrix_size * matrix_size);
  }

  void init_inputs() {
    std::cout<< " data input init ...." << std::endl;

    for (int i = 0; i < matrix_size * matrix_size; i++) {
      auto a = static_cast<float>(std::rand()) / RAND_MAX;
      auto b = static_cast<float>(std::rand()) / RAND_MAX;

      //TODO: 乘以matrix_size的意义？
      src_A[i] = static_cast<TYPE>(a * matrix_size);
      src_B[i] = static_cast<TYPE>(b * matrix_size);
    }
  }


  // 向A B 存数据? 但是
  void matmul_cpu() {
    for (int row = 0 ； row < matrix_size; row++) {
      for (int col = 0; col < matrix_size; col++) {
        TYPE sum = 0;
        // dot product
        for (int k = 0; k < matrix_size; k++) {
          sum += src_A[row * matrix_size + k] * src_B[k * matrix_size + col];
        }
        refs[row * matrix_size + col] = sum;
      }
    }
  }



  std::vector<TYPE> src_A;
  std::vector<TYPE> src_B;
  std::vector<TYPE> refs;

  std::vector<uint8_t> A_mat;
  std::vector<uint8_t> B_mat;

private:
  uint32_t buf_size;
  uint32_t matrix_size;
};

int main(int argc, char *argv[]) {
  uint32_t data_size = 0;

  parse_args(argc, argv, data_size);

  if (matrix_size == 0) {
    std::cout << "Matrix size must be greater than 0." << std::endl;
    return -1;
  }

  // 打开gpgpu设备端
  std::cout << "open device connection" << std::endl;
  RT_CHECK(vx_dev_open(&device));

  // 硬件配置 + tensor core配置初始化
  uint64_t num_cores, num_warps, num_threads;
  uint64_t tc_size, tc_per_warp;
  // 每个tensor core分配给多少个thread使用
  int threads_per_tc;

  // 从设备读取配置
  RT_CHECK(vx_dev_caps(device, VX_CAPS_NUM_CORES, &num_cores));
  RT_CHECK(vx_dev_caps(device, VX_CAPS_NUM_WARPS, &num_warps));
  RT_CHECK(vx_dev_caps(device, VX_CAPS_NUM_THREADS, &num_threads));
  RT_CHECK(vx_dev_caps(device, VX_CAPS_TC_SIZE, &tc_size));
  RT_CHECK(vx_dev_caps(device, VX_CAPS_TC_NUM, &tc_per_warp));

  // 如果每个warp里面的tensor core数量比线程还多，那么每个tensor对应一个thread还有多余的
  if (tc_per_warp > num_threads) {
    threads_per_tc = 1;
  } else {
    // thread总数 / 每个warp共用的tensor core数
    threads_per_tc = num_threads / tc_per_warp; // TODO:这里的设计是整除，如果是11个thread，3 tensor core，是设计为 5 5 1？还是 4 4 3？ 平均分配比较好
  }

  // 每个tensor_core的 thread num x 运算次数 , 但是还没考虑有多少个warp?
  // 举例 每个tensor core 上面对应 2 thread， tc size = 4, matrix_size = 16, 那么每个tensor core上面要运算的总次数是 2 * 4 * 4 = 32

  uint32_t tiles_num = (matrix_size * matrix_size) / (tc_size * tc_size);

  uint32_t num_tasks = tiles_num * threads_per_tc;

  // 大小要考虑 operand collector的微架构
  uint32_t buf_size = tiles_num * (matrix_size / tc_size) * (tc_size * tc_size) * data_size;

  std::cout << "Debug :: buf size = " << buf_size << std::endl;

  std::cout << "Allocating buffers" << std::endl;

  RT_CHECK(vx_mem_alloc(device, buf_size, VX_MEM_READ, &A_buffer));
  RT_CHECK(vx_mem_address(A_buffer, &kernel_arg.src0_addr));

  RT_CHECK(vx_mem_alloc(device, buf_size, VX_MEM_READ, &B_buffer));
  RT_CHECK(vx_mem_address(B_buffer, &kernel_arg.src1_addr));

  RT_CHECK(vx_mem_alloc(device, buf_size, VX_MEM_WRITE, &C_buffer));
  RT_CHECK(vx_mem_address(C_buffer, &kernel_arg.dst_addr));

  // 打印分配的设备内存地址
  std::cout << "A_addr=0x" << std::hex << kernel_arg.src0_addr << std::endl;
  std::cout << "B_addr=0x" << std::hex << kernel_arg.src1_addr << std::endl;
  std::cout << "C_addr=0x" << std::hex << kernel_arg.dst_addr << std::endl;

  // 管理CPU端数据
  mainVars<int> variables (buf_size, matrix_size);
  variables.init_inputs();

  // generate matmul data
  variables.matmul_cpu();

  // 一个tensor core的元素总数
  uint32_t tc_size_f = tc_size * tc_size;

  uint32_t n_tiles = matrix_size / tc_size;
  //TODO: 这里的规定是 matrix_size 必须是 tc_size整数倍？
  if (matrix_size % tc_size != 0) {
    std::cerr << "Matrix size must be a multiple of TC size." << std::endl;
    cleanup();
    return -1;
  }


  variables.A_mat.resize(buf_size);
  variables.B_mat.resize(buf_size);

  for (int k = 0; k < n_tiles; k++) {
    for (int i = 0; i < n_tiles; i++) {
      for (int j = 0; j < n_tiles; j++) {
        // 一个tensor core的block
        for (int t  = 0; t < tc_size * tc_size; t++) {
          auto index = n_tiles * n_tiles * tc_size_f * k + n_tiles * tc_size_f * i + tc_size_f * j + t;
          variables.A_mat[index] = variables.src_A[k*tc_size*matrix_size+ tc_size*j +(t/tc_size)*matrix_size + t%tc_size];
        }
      }
    }
  }
}