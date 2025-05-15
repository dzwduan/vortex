_start:
  # 初始化每个线程的寄存器
  csrr  t0, VX_CSR_NUM_WARPS  # 读取自定义 CSR，获取 Warp 的数量到寄存器 t0
  la    t1, init_regs_all   # 加载 init_regs_all 函数的地址到寄存器 t1
  # .insn 指令用于直接指定指令编码，这里调用自定义指令 0 (RISCV_CUSTOM0)
  # 语法是 wspawn <warp_count_reg>, <function_ptr_reg>
  # 让所有 Warp (t0 个) 开始执行 init_regs_all 函数
  .insn r RISCV_CUSTOM0, 1, 0, x0, t0, t1  # wspawn t0, t1

  # 以下几行确保当前 Warp (执行 _start 的 Warp) 也执行初始化
  li    t0, -1              # 将 -1 (所有位为1) 加载到 t0，用作线程掩码
  # 调用自定义指令 0，tmc (thread mask control)
  # 激活当前 Warp 内的所有线程 (使用 -1 掩码)
  .insn r RISCV_CUSTOM0, 0, 0, x0, t0, x0  # tmc t0
  jal   init_regs           # 跳转并链接到 init_regs 函数，为当前 Warp 的所有活动线程初始化寄存器
  li    t0, 1               # 将 1 加载到 t0 (掩码，只激活线程 0)
  # 意图：只保留当前 Warp 的线程 0 活动，用于执行后续的串行代码
  .insn r RISCV_CUSTOM0, 0, 0, x0, t0, x0  # tmc t0

  # 为所有 Warp 初始化 TLS (线程局部存储) - 逻辑与寄存器初始化类似
  csrr  t0, VX_CSR_NUM_WARPS
  la    t1, init_tls_all
  .insn r RISCV_CUSTOM0, 1, 0, x0, t0, t1  # wspawn t0, t1 (让所有 Warp 执行 init_tls_all)
  li    t0, -1
  .insn r RISCV_CUSTOM0, 0, 0, x0, t0, x0  # tmc t0 (当前 Warp 激活所有线程)
  call  __init_tls          # 调用 C 库或运行时提供的 TLS 初始化函数
  li    t0, 1
  .insn r RISCV_CUSTOM0, 0, 0, x0, t0, x0  # tmc t0 (当前 Warp 只保留线程 0 活动)

  # 清零 BSS 段 (未初始化的全局变量和静态变量)
  la    a0, _edata          # a0 = BSS 段的起始地址 (_edata 是 data 段的结束地址)
  la    a2, _end            # a2 = BSS 段的结束地址
  sub   a2, a2, a0          # a2 = BSS 段的大小 (长度)
  li    a1, 0               # a1 = 要填充的值 (0)
  call  memset            # 调用 memset(start_addr, value, length) 清零 BSS

  # 初始化陷阱(中断/异常)向量 (这部分被注释掉了)
  # la t0, trap_entry       # 加载陷阱处理入口函数的地址
  # csrw mtvec, t0          # 将地址写入 mtvec (Machine Trap Vector) CSR

#ifdef HAVE_INITFINI_ARRAY
  # 如果定义了这个宏 (通常在支持 C++ 全局对象构造/析构时) , 这块存在于syscall
  # 则运行全局初始化函数 (例如 C++ 的全局构造函数)
  call  __libc_init_array
#endif

  # 调用主程序例程
  call  main                # 调用 C/C++ 的 main 函数

  # 调用退出例程
  tail  exit                # main 函数返回后，使用 tail call (尾调用优化) 跳转到 exit 函数
                            # tail exit 效果类似 j exit，并且会优化栈帧
.size _start, .-_start      # 标记 _start 函数的大小