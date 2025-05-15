#include "tests.h"
#include <stdio.h>
#include <algorithm>
#include <VX_config.h>
#include <vx_intrinsics.h>
#include <vx_print.h>
#include <vx_spawn.h>

int __attribute__((noinline)) check_error(const int* buffer, int offset, int size) {
	int errors = 0;
	for (int i = offset; i < size; i++)	{
		int value = buffer[i];
		int ref_value = 65 + i;
		if (value != ref_value)	{
			PRINTF(ANSI_COLOR_RED "error: [%d] 0x%x, expected 0x%x" ANSI_COLOR_GREEN "\n", i, value, ref_value);
			++errors;
		}
	}
	return errors;
}

int __attribute__((noinline)) make_select_tmask(int tid) {
	return (1 << tid);
}

int __attribute__((noinline)) make_full_tmask(int num_threads) {
	return (1 << num_threads) - 1;
}

///////////////////////////////////////////////////////////////////////////////

#define GLOBAL_MEM_SZ 8
int global_buffer[GLOBAL_MEM_SZ];

// 理解为分配在全局内存上的，自己申请一个空间就行
int test_global_memory() {
	PRINTF("Global Memory Test\n");

	for (int i = 0; i < GLOBAL_MEM_SZ; i++) {
		global_buffer[i] = 65 + i;
	}

	return check_error(global_buffer, 0, GLOBAL_MEM_SZ);
}

///////////////////////////////////////////////////////////////////////////////

// volatile 代表禁用优化 ，lmem_addr绑定了硬件的base_addr ，我关了没啥事
volatile int* lmem_addr = (int*)LMEM_BASE_ADDR;

int lmem_buffer[8];

// noinline 禁用内联优化
void __attribute__((noinline)) do_lmem_wr() {
	unsigned tid = vx_thread_id();
	// 下面三行貌似是测试RAW , 删了两行也没事
	lmem_addr[tid] = 65 + tid;
	// int x = lmem_addr[tid];
	// lmem_addr[tid] = x;
}

void __attribute__((noinline)) do_lmem_rd() {
	unsigned tid = vx_thread_id();
	lmem_buffer[tid] = lmem_addr[tid];
}

int test_local_memory() {
	PRINTF("Local Memory Test\n");

	int num_threads = std::min(vx_num_threads(), 8);
	int tmask = make_full_tmask(num_threads);
	// 启用所有线程
	vx_tmc(tmask);
	do_lmem_wr();
	do_lmem_rd();
	// 切换到从thread 0 开始执行
	vx_tmc_one();

	return check_error(lmem_buffer, 0, num_threads);
}

///////////////////////////////////////////////////////////////////////////////

int tmc_buffer[8];

void __attribute__((noinline)) do_tmc() {
	unsigned tid = vx_thread_id();
	tmc_buffer[tid] = 65 + tid;
}

int  test_tmc() {
	PRINTF("TMC Test\n");

	int num_threads = std::min(vx_num_threads(), 8);
	int tmask = make_full_tmask(num_threads);
	// set thread mask
	vx_tmc(tmask);
	// 根据 thread mask 写入 tmc_buffer
	do_tmc();
	// 执行，执行会根据tmc mask来决定执行哪些线程？这个怎么看代码体现?
	vx_tmc_one();

	return check_error(tmc_buffer, 0, num_threads);
}

///////////////////////////////////////////////////////////////////////////////

int pred_buffer[8];

void __attribute__((noinline)) do_pred() {
	unsigned tid = vx_thread_id();
	vx_pred((tid == 0), 1);
	pred_buffer[tid] = 65;
}

int test_pred() {
	PRINTF("PRED Test\n");
	int num_threads = std::min(vx_num_threads(), 8);
	int tmask = make_full_tmask(num_threads);

	for (int i = 1; i < num_threads; i++) {
		pred_buffer[i] = 65 + i;
	}

	vx_tmc(tmask);
	do_pred();
	vx_tmc_one();

	return check_error(pred_buffer, 0, num_threads);
}

///////////////////////////////////////////////////////////////////////////////

int wspawn_buffer[8];

void wspawn_kernel() {
	unsigned wid = vx_warp_id();
	wspawn_buffer[wid] = 65 + wid;
	vx_tmc(0 == wid);
}
// 以warp为粒度，是先vx_wspawn，然后执行wspawn_kernel
int test_wsapwn() {
	PRINTF("Wspawn Test\n");
	int num_warps = std::min(vx_num_warps(), 8);
	// 多少个 warp执行wspawn_kernel
	vx_wspawn(num_warps, wspawn_kernel);
	
	wspawn_kernel();

	//TODO: 这里没有tmc_one?

	return check_error(wspawn_buffer, 0, num_warps);
}

///////////////////////////////////////////////////////////////////////////////

int dvg_buffer[4];

//每次遇到条件分支，都需要加上vx_split和vx_join
void __attribute__((noinline)) __attribute__((optimize("O1"))) do_divergence() {
	int tid = vx_thread_id();
	int cond1 = tid < 2;
	int sp1 = vx_split(cond1);
	if (cond1) {
		{
			int cond2 = tid < 1;
			int sp2 = vx_split(cond2);
			if (cond2) {
				dvg_buffer[tid] = 65; // A thread 0
			} else {
				dvg_buffer[tid] = 66; // B thread 1
			}
			vx_join(sp2);
		}
		{
			int cond3 = tid < 0;
			int sp3 = vx_split(cond3);
			if (cond3) {
				dvg_buffer[tid] = 67; // C thread 2
			}
			vx_join(sp3);
		}
	} else {
		{
			int cond2 = tid < 3;
			int sp2 = vx_split(cond2);
			if (cond2) {
				dvg_buffer[tid] = 67; // C thread 3
			} else {
				dvg_buffer[tid] = 68; // D thread 4
			}
			vx_join(sp2);
		}
	}
	// 重聚点
	vx_join(sp1);
}

int test_divergence() {
	PRINTF("Control Divergence Test\n");

	int num_threads = std::min(vx_num_threads(), 4);
	int tmask = make_full_tmask(num_threads);
	vx_tmc(tmask);
	do_divergence();
	vx_tmc_one();

	return check_error(dvg_buffer, 0, num_threads);
}

///////////////////////////////////////////////////////////////////////////////

#define ST_BUF_SZ 8
typedef struct {
	int * src;
	int * dst;
} st_args_t;

int st_buffer_src[ST_BUF_SZ];
int st_buffer_dst[ST_BUF_SZ];

void st_kernel(const st_args_t * __UNIFORM__ arg) {
  arg->dst[blockIdx.x] = arg->src[blockIdx.x];
}

int test_spawn_tasks() {
	PRINTF("SpawnTasks Test\n");

	st_args_t arg;
	arg.src = st_buffer_src;
	arg.dst = st_buffer_dst;

	for (int i = 0; i < ST_BUF_SZ; i++) {
		st_buffer_src[i] = 65 + i;
	}

	uint32_t num_tasks = ST_BUF_SZ;
	vx_spawn_threads(1, &num_tasks, nullptr, (vx_kernel_func_cb)st_kernel, &arg);

	return check_error(st_buffer_dst, 0, ST_BUF_SZ);
}

///////////////////////////////////////////////////////////////////////////////

typedef struct {
	int * buf;
} sr_args_t;

int sr_buffer[8];

void sr_kernel(const sr_args_t * arg) {
	int tid = vx_thread_id();
  	arg->buf[tid] = 65 + tid;
}

void __attribute__((noinline)) do_serial() {
	sr_args_t arg;
	arg.buf = sr_buffer;
	// vx_serial按照线程id串行执行给定的callback
	// 注意最后仍然要调用vx_tmc_one来进行真正的执行操作
	vx_serial((vx_serial_cb)sr_kernel, &arg);
}

int test_serial() {
	PRINTF("Serial Test\n");
	int num_threads = std::min(vx_num_threads(), 8);
	int tmask = make_full_tmask(num_threads);
	vx_tmc(tmask);
	do_serial();
	vx_tmc_one();

	return check_error(sr_buffer, 0, num_threads);
}

///////////////////////////////////////////////////////////////////////////////

int tmask_buffer[8];

int __attribute__((noinline)) do_tmask() {
	int tid = vx_thread_id();
	int tmask = make_select_tmask(tid);
	int cur_tmask = vx_active_threads();
	tmask_buffer[tid] = (cur_tmask == tmask) ? (65 + tid) : 0;
	return tid + 1;
}

int test_tmask() {
	PRINTF("Thread Mask Test\n");

	// activate all thread
	vx_tmc(-1);

	int num_threads = std::min(vx_num_threads(), 8);
	int tid = 0;

l_start:
	int tmask = make_select_tmask(tid);
	vx_tmc(tmask);
	tid = do_tmask();
	if (tid < num_threads)
		goto l_start;

	vx_tmc_one();

	return check_error(tmask_buffer, 0, num_threads);
}

///////////////////////////////////////////////////////////////////////////////

int barrier_buffer[8];
volatile int barrier_ctr;
volatile int barrier_stall;

void barrier_kernel() {
	unsigned wid = vx_warp_id();
	for (int i = 0; i <= (wid * 256); ++i) {
		++barrier_stall;
	}
	barrier_buffer[wid] = 65 + wid;
	vx_barrier(0, barrier_ctr);
	// vx_printf("wid = %d\n", wid);
	vx_tmc(0 == wid);  // 只处理第0个
}

int test_barrier() {
	PRINTF("Barrier Test\n");
	int num_warps = std::min(vx_num_warps(), 8);
	barrier_ctr = num_warps;
	barrier_stall = 0;
	// vx_wspawn + callback_func 需要配合使用
	vx_wspawn(num_warps, barrier_kernel);
	barrier_kernel();


	return check_error(barrier_buffer, 0, num_warps);
}

///////////////////////////////////////////////////////////////////////////////

int tls_buffer[8];
__thread int tls_var;

__attribute__((noinline)) void print_tls_var() {
	unsigned wid = vx_warp_id();
	tls_buffer[wid] = 65 + tls_var;
}

void tls_kernel() {
	unsigned wid = vx_warp_id();
	tls_var = wid;
	print_tls_var();
	vx_tmc(0 == wid);
}

int test_tls() {
	PRINTF("TLS Test\n");
	int num_warps = std::min(vx_num_warps(), 8);
	vx_wspawn(num_warps, tls_kernel);
	tls_kernel();
	return check_error(tls_buffer, 0, num_warps);
}