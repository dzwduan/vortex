// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <vx_spawn.h>
#include <vx_intrinsics.h>
#include <vx_print.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

__thread dim3_t blockIdx;
__thread dim3_t threadIdx;
dim3_t gridDim;
dim3_t blockDim;

__thread uint32_t __local_group_id;
uint32_t __warps_per_group;

typedef struct {
	vx_kernel_func_cb callback;
	const void* arg;
	uint32_t group_offset;
	uint32_t warp_batches;
	uint32_t remaining_warps;
  uint32_t warps_per_group;
  uint32_t groups_per_core;
  uint32_t remaining_mask;
} wspawn_groups_args_t;

typedef struct {
	vx_kernel_func_cb callback;
	const void* arg;
	uint32_t all_tasks_offset;
  uint32_t remain_tasks_offset;
	uint32_t warp_batches;
	uint32_t remaining_warps;
} wspawn_threads_args_t;

static void __attribute__ ((noinline)) process_threads() {
  wspawn_threads_args_t* targs = (wspawn_threads_args_t*)csr_read(VX_CSR_MSCRATCH);

  uint32_t threads_per_warp = vx_num_threads();
  uint32_t warp_id = vx_warp_id();
  uint32_t thread_id = vx_thread_id();

  uint32_t start_warp = (warp_id * targs->warp_batches) + MIN(warp_id, targs->remaining_warps);
  uint32_t iterations = targs->warp_batches + (warp_id < targs->remaining_warps);

  uint32_t start_task_id = targs->all_tasks_offset + (start_warp * threads_per_warp) + thread_id;
  uint32_t end_task_id = start_task_id + iterations * threads_per_warp;

  __local_group_id = 0;
  threadIdx.x = 0;
  threadIdx.y = 0;
  threadIdx.z = 0;

  vx_kernel_func_cb callback = targs->callback;
  const void* arg = targs->arg;

  for (uint32_t task_id = start_task_id; task_id < end_task_id; task_id += threads_per_warp) {
    blockIdx.x = task_id % gridDim.x;
    blockIdx.y = (task_id / gridDim.x) % gridDim.y;
    blockIdx.z = task_id / (gridDim.x * gridDim.y);
    callback((void*)arg);
  }
}

static void __attribute__ ((noinline)) process_remaining_threads() {
  wspawn_threads_args_t* targs = (wspawn_threads_args_t*)csr_read(VX_CSR_MSCRATCH);

  uint32_t thread_id = vx_thread_id();
  uint32_t task_id = targs->remain_tasks_offset + thread_id;

  (targs->callback)((void*)targs->arg);
}

static void __attribute__ ((noinline)) process_threads_stub() {
  // activate all threads
  vx_tmc(-1);

  // process all tasks
  process_threads();

  // disable warp
  vx_tmc_zero();
}

static void __attribute__ ((noinline)) process_thread_groups() {
  wspawn_groups_args_t* targs = (wspawn_groups_args_t*)csr_read(VX_CSR_MSCRATCH);

  uint32_t threads_per_warp = vx_num_threads();
  uint32_t warp_id = vx_warp_id();
  uint32_t thread_id = vx_thread_id();

  uint32_t warps_per_group = targs->warps_per_group;
  uint32_t groups_per_core = targs->groups_per_core;

  uint32_t iterations = targs->warp_batches + (warp_id < targs->remaining_warps);

  uint32_t local_group_id = warp_id / warps_per_group;
  uint32_t group_warp_id = warp_id - local_group_id * warps_per_group;
  uint32_t local_task_id = group_warp_id * threads_per_warp + thread_id;

  uint32_t start_group = targs->group_offset + local_group_id;
  uint32_t end_group = start_group + iterations * groups_per_core;

  __local_group_id = local_group_id;

  threadIdx.x = local_task_id % blockDim.x;
  threadIdx.y = (local_task_id / blockDim.x) % blockDim.y;
  threadIdx.z = local_task_id / (blockDim.x * blockDim.y);

  vx_kernel_func_cb callback = targs->callback;
  const void* arg = targs->arg;

  for (uint32_t group_id = start_group; group_id < end_group; group_id += groups_per_core) {
    blockIdx.x = group_id % gridDim.x;
    blockIdx.y = (group_id / gridDim.x) % gridDim.y;
    blockIdx.z = group_id / (gridDim.x * gridDim.y);
    callback((void*)arg);
  }
}

static void __attribute__ ((noinline)) process_thread_groups_stub() {
  wspawn_groups_args_t* targs = (wspawn_groups_args_t*)csr_read(VX_CSR_MSCRATCH);
  uint32_t warps_per_group = targs->warps_per_group;
  uint32_t remaining_mask = targs->remaining_mask;
  uint32_t warp_id = vx_warp_id();
  uint32_t group_warp_id = warp_id % warps_per_group;
  uint32_t threads_mask = (group_warp_id == warps_per_group-1) ? remaining_mask : -1;

  // activate threads
  vx_tmc(threads_mask);

  // process thread groups
  process_thread_groups();

  // disable all warps except warp0
  vx_tmc(0 == vx_warp_id());
}

int vx_spawn_threads(uint32_t dimension,            // 输入参数：维度数量 (例如 1D, 2D, 3D)
                     const uint32_t* grid_dim,      // 输入参数：指向包含每个维度网格大小的数组指针
                     const uint32_t* block_dim,     // 输入参数：指向包含每个维度块大小的数组指针
                     vx_kernel_func_cb kernel_func, // 输入参数：内核函数的回调指针
                     const void* arg) {             // 输入参数：传递给内核函数的参数

  // 1. 计算线程组数量和线程组大小 
  uint32_t num_groups = 1; // 线程组（通常称为“块”或工作组）的总数
  uint32_t group_size = 1; // 每个线程组中的线程数
  for (uint32_t i = 0; i < 3; ++i) { // 遍历最多三个维度 (X, Y, Z)
    uint32_t gd = (grid_dim && (i < dimension)) ? grid_dim[i] : 1; // 当前维度的网格大小
    uint32_t bd = (block_dim && (i < dimension)) ? block_dim[i] : 1; // 当前维度的块大小
    num_groups *= gd;   // 累乘得到总的线程组数量
    group_size *= bd;   // 累乘得到每个线程组内的线程总数
    gridDim.m[i] = gd;  // 存储计算出的网格维度信息 (假设 gridDim 是全局或某个结构体成员)
    blockDim.m[i] = bd; // 存储计算出的块维度信息 (假设 blockDim 是全局或某个结构体成员)
  }

  // 2. 获取设备规格参数
  // 这些函数很可能查询硬件寄存器或运行时系统来获取设备属性
  uint32_t num_cores = vx_num_cores();         // 可用的处理核心总数
  uint32_t warps_per_core = vx_num_warps();      // 单个核心可以管理/并发执行的 Warp 数量
  uint32_t threads_per_warp = vx_num_threads();  // 单个 Warp 中的线程数量 (一种常见的 SIMD 执行单元)
  uint32_t core_id = vx_core_id();             // 获取当前执行此函数的物理核心的 ID

  // 3. 检查线程组大小的合法性
  uint32_t threads_per_core = warps_per_core * threads_per_warp; // 单个核心能处理的最大线程数
  if (threads_per_core < group_size) {
    // 如果请求的线程组中的线程数超过了单个核心的处理能力，则报错
    vx_printf("error: group_size > threads_per_core (%d,%d)\n", group_size, threads_per_core);
    return -1; // 返回错误码
  }

  // 4. 主要执行逻辑：根据 group_size 分为两种路径
  if (group_size > 1) { // 路径1：处理多线程的线程组 (典型的并行执行模式)
    // 4.a. 计算每个线程组需要的 Warp 配置
    uint32_t warps_per_group = group_size / threads_per_warp; // 一个线程组需要多少个完整的 Warp
    uint32_t remaining_threads = group_size % threads_per_warp; // 剩余的、不足以填满一个 Warp 的线程
    uint32_t remaining_mask = (uint32_t)-1; // 用于最后一个、可能不完整的 Warp 的线程掩码。-1 通常表示所有线程都激活。
    if (remaining_threads != 0) {
      remaining_mask = (1 << remaining_threads) - 1; // 为不完整的 Warp 中的活动线程创建一个位掩码
      ++warps_per_group; // 因为有剩余线程，所以需要额外一个 Warp
    }

    // 4.b. 计算执行所有线程组所必需的活动核心数量
    uint32_t needed_warps = num_groups * warps_per_group; // 所有线程组总共需要的 Warp 数量
    // 向上取整除法: (needed_warps + warps_per_core - 1) / warps_per_core
    uint32_t needed_cores = (needed_warps + warps_per_core - 1) / warps_per_core; // 处理所有 Warp 所需的最小核心数
    uint32_t active_cores = MIN(needed_cores, num_cores); // 使用所需的核心数，但不能超过实际可用的核心数

    // 4.c. 核心参与检查
    // 如果当前核心的 ID 大于或等于活动核心的数量，则此核心不参与此次内核启动，直接返回。
    // 这是 SPMD (单指令多数据) 编程模型中的常见模式，所有核心都运行这段设置代码，但只有一部分核心继续执行实际工作。
    if (core_id >= active_cores)
      return 0; // 此核心在此次内核启动中不活动

    // 4.d. 在活动核心之间分配线程组
    uint32_t total_groups_per_core = num_groups / active_cores; // 每个活动核心平均分配到的线程组数量
    uint32_t remaining_groups_to_distribute = num_groups % active_cores; // 平均分配后剩余的线程组数量
    if (core_id < remaining_groups_to_distribute) {
      ++total_groups_per_core; // ID 较小的前 'remaining_groups_to_distribute' 个核心多分配一个线程组
    }

    // 4.e. 计算当前核心上需要激活的 Warp 数量 (可能分批次处理)
    // 这部分逻辑似乎处理当一个核心无法同时启动其分配到的所有线程组的 Warp 时的情况。
    uint32_t groups_launchable_concurrently_on_core = warps_per_core / warps_per_group; // 此核心能同时运行多少个完整线程组的 Warp
    uint32_t total_warps_for_this_core = total_groups_per_core * warps_per_group; // 此核心需要管理的总 Warp 数
    uint32_t active_warps_this_batch = total_warps_for_this_core; // 当前批次要激活的 Warp 数 (初始为全部)
    uint32_t warp_batches = 1; // 此核心将处理的 Warp 批次数
    uint32_t remaining_warps_after_batches = 0; // 最后一批（可能较小）中的 Warp 数

    if (active_warps_this_batch > warps_per_core) { // 如果此核心需要管理的 Warp 数超过了它能一次运行的数量
      active_warps_this_batch = groups_launchable_concurrently_on_core * warps_per_group; // 激活尽可能多的完整线程组的 Warp
      warp_batches = total_warps_for_this_core / active_warps_this_batch; // 计算完整批次的数量
      remaining_warps_after_batches = total_warps_for_this_core % active_warps_this_batch; // 最后一批（可能不完整）的 Warp 数
    }

    // 4.f. 计算此核心处理的线程组的偏移量 (起始索引)
    // `total_groups_per_core` 已经包含了处理余数的核心会多一个组的逻辑。
    // `remaining_groups_per_core` 应该是 `num_groups % active_cores`。
    // 这个偏移量决定了当前核心从哪个全局线程组索引开始处理。
    uint32_t group_offset = core_id * (num_groups / active_cores) + MIN(core_id, num_groups % active_cores);
    // 代码中的 `group_offset = core_id * total_groups_per_core + MIN(core_id, remaining_groups_per_core);`
    // 其中 `remaining_groups_per_core` 是 `num_groups - active_cores * total_groups_per_core`。
    // 如果 `total_groups_per_core` 是向下取整的 `num_groups / active_cores`，那么 `remaining_groups_per_core` 就是 `num_groups % active_cores`。
    // 鉴于 `total_groups_per_core` 可能已经加了1，这里的计算可能依赖特定的分配策略，但目标是为每个核心找到正确的起始组。

    // 4.g. 设置调度器参数
    // 准备一个结构体，包含 Warp 派生/调度机制所需的所有信息。
    wspawn_groups_args_t wspawn_args = {
      kernel_func,                   // 指向用户内核函数的指针
      arg,                           // 内核函数的参数
      group_offset,                  // 此核心的 Warp 处理的起始线程组索引
      warp_batches,                  // 此核心将处理的 Warp 批次数
      remaining_warps_after_batches, // 最后一批中的 Warp 数
      warps_per_group,               // 单个线程组所需的 Warp 数
      groups_launchable_concurrently_on_core, // 此核心能并发运行多少个线程组的 Warp
      remaining_mask                 // 线程组最后一个 Warp (若不完整) 的线程掩码
    };
    csr_write(VX_CSR_MSCRATCH, &wspawn_args); // 将这些参数写入特定的 CSR (机器暂存寄存器)
                                              // 这是将复杂参数传递给底层陷阱处理程序或硬件调度的常用方法。

    // 4.h. 设置全局变量 (可能供存根函数读取)
    __warps_per_group = warps_per_group; // 将 warps_per_group 存储为全局变量，供存根函数访问

    // 4.i. 派生 (Spawn) Warp
    // `vx_wspawn` 很可能在其他 Warp 上启动执行，或通知硬件调度器这样做。
    // 参数 `active_warps_this_batch` 表示总共需要激活多少 Warp 来执行 `process_thread_groups_stub`。
    vx_wspawn(active_warps_this_batch, process_thread_groups_stub); // 派生其他 Warp 来运行线程组处理存根

    // 4.j. 在当前 Warp (此核心的 Warp 0) 上执行
    process_thread_groups_stub(); // 当前 Warp 也运行存根函数。
                                  // 这个存根函数将使用来自 VX_CSR_MSCRATCH 的参数来处理其分配到的线程组。

  } else { // 路径2：group_size == 1 (或 <=1)，实际上是单线程任务
    // 此分支处理每个“线程组”只是一个单独任务/线程的情况。
    // 逻辑与多线程组情况类似，但针对单个任务进行了简化。

    uint32_t num_tasks = num_groups; // 此时 num_groups 代表单个任务的总数
    __warps_per_group = 0; // 当 group_size 为 1 时，没有“每组 Warp 数”的概念

    // 4.k. 计算执行所有任务所需的活动核心数量
    uint32_t needed_cores = (num_tasks + threads_per_core - 1) / threads_per_core; // 如果每个任务是一个线程，则需要的核心数
    uint32_t active_cores = MIN(needed_cores, num_cores);

    // 4.l. 核心参与检查
    if (core_id >= active_cores)
      return 0; // 此核心不活动

    // 4.m. 在活动核心之间分配任务
    uint32_t tasks_per_core = num_tasks / active_cores; // 每个活动核心平均分配到的任务数
    uint32_t remaining_tasks_to_distribute_across_cores = num_tasks % active_cores; // 平均分配后剩余的任务数
    if (core_id < remaining_tasks_to_distribute_across_cores) {
      ++tasks_per_core; // ID 较小的前 'remaining_tasks_to_distribute_across_cores' 个核心多分配一个任务
    }

    // 4.n. 计算当前核心上为任务需要激活的 Warp 数量
    // 每个任务是一个线程。计算这些任务将填满多少 Warp。
    uint32_t total_warps_for_this_core_tasks = tasks_per_core / threads_per_warp; // 此核心上任务所需的完整 Warp 数
    uint32_t remaining_individual_tasks = tasks_per_core % threads_per_warp;    // 此核心上不足以填满一个 Warp 的剩余任务数

    uint32_t active_warps_this_batch_tasks = total_warps_for_this_core_tasks;
    uint32_t warp_batches_tasks = 1;
    uint32_t remaining_warps_after_batches_tasks = 0;

    if (active_warps_this_batch_tasks > warps_per_core) { // 如果需要的 Warp 数超过核心一次能运行的数量
      active_warps_this_batch_tasks = warps_per_core; // 最大化一个批次中使用的 Warp 数
      warp_batches_tasks = total_warps_for_this_core_tasks / active_warps_this_batch_tasks;
      remaining_warps_after_batches_tasks = total_warps_for_this_core_tasks % active_warps_this_batch_tasks;
    }

    // 4.o. 计算此核心处理的任务的偏移量
    // 此核心处理的所有任务的起始索引
    uint32_t all_tasks_offset_for_core = core_id * (num_tasks / active_cores) + MIN(core_id, num_tasks % active_cores);
    // 将由剩余线程（非完整 Warp）处理的任务的起始索引
    uint32_t remain_tasks_offset_for_core = all_tasks_offset_for_core + (tasks_per_core - remaining_individual_tasks);

    // 4.p. 准备任务的调度器参数
    wspawn_threads_args_t wspawn_args = { // 用于基于任务派生的不同参数结构
      kernel_func,
      arg,
      all_tasks_offset_for_core,        // 此核心处理的所有任务的起始索引
      remain_tasks_offset_for_core,     // （非完整 Warp 的）剩余独立任务的起始索引
      warp_batches_tasks,
      remaining_warps_after_batches_tasks
    };
    csr_write(VX_CSR_MSCRATCH, &wspawn_args); // 写入 CSR

    // 4.q. 执行任务
    if (active_warps_this_batch_tasks >= 1) { // 如果有任何完整的 Warp 需要启动
      // 派生其他 Warp 来运行任务处理存根
      vx_wspawn(active_warps_this_batch_tasks, process_threads_stub);

      vx_tmc(-1); // `vx_tmc` 很可能意为 "Thread Mask Control" (线程掩码控制)。-1 可能表示激活当前 Warp 中的所有线程。

      process_threads(); // 当前 Warp (Warp 0) 处理其分配到的 (完整 Warp 的) 任务。
                         // 此函数可能迭代 `threads_per_warp` 次，为每个任务调用 kernel_func。

      vx_tmc_one(); // 将线程掩码重置为仅当前 Warp 中的线程 0 活动。
    }

    if (remaining_individual_tasks != 0) { // 如果有剩余的、未形成完整 Warp 的任务
      uint32_t tmask = (1 << remaining_individual_tasks) - 1; // 为剩余任务数量创建掩码
      vx_tmc(tmask); // 仅激活当前 Warp 中处理这些剩余任务所需的线程。

      process_remaining_threads(); // 处理这些剩余的任务。

      vx_tmc_one(); // 重置线程掩码。
    }
  }

  // 5. 等待完成
  // `vx_wspawn(1, 0)` 或 `vx_wspawn(0,0)` 通常是一个屏障/同步调用。
  // 它可能意味着“等待所有先前派生的 Warp (在此核心上或全局) 完成”。
  // '1' 可能表示这是一个与主 Warp/线程相关的等待。
  // 函数指针为 '0' 表示不调用新函数，仅等待。
  vx_wspawn(1, 0);

  return 0; // 成功
}

#ifdef __cplusplus
}
#endif
