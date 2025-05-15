#ifndef TESTS
#define TESTS

#define PRINTF vx_printf

#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_RESET   "\x1b[0m"

int test_global_memory();

int test_local_memory();

int test_tmc();

int test_pred();

int test_divergence();

int test_wsapwn();

int test_spawn_tasks();

int test_serial();

int test_tmask();

int test_barrier();

int test_tls();

#endif
