## Structure



```
benchmark/src/
├── README.md
├── c
├── launcher
├── main
├── support
└── triton
```

Here is the structure of source code. 
`main` includes each main cpp code for the whole application, which involves input/output data initialization(read/write files), calling kernel, time and so on. `support` includes some basic functions used by `main`. `c` and `triton` should contains C or Triton version benchmark source code.

There is a directory `launcher` for Triton. It includes `include` and  `src` two sub-directories. For each header file in `launcher/include`, it has a function pointer(say, `kernel_ptr_t`) of Triton kernel, which is in the `extern "C"{}`, and a wrap function, which provides necessary parameters for `kernel_ptr_t`. You can see the implementation of wrap functions in `launcher/src`. Then, the cpp file in the `main`, it just call that wrap function to launch Triton kernel.

In fact, these two kinds of files under `launcher/include` and `launcher/src` are generated when we are running single Triton programme(e.g. `KERNEL_LAUNCHER_INCLUDE_DIR=${KERNEL_LAUNCHER_INCLUDE_DIR} KERNEL_AUX_FILE_DIR=${KERNEL_AUX_FILE_DIR} TRITON_CPU_BACKEND=1 python3 <matmul.py>`), and the Triton compiler(you can check that patch file for more details if interested) would write those two kinds of files in the specified locations(`${KERNEL_LAUNCHER_INCLUDE_DIR}` and `${KERNEL_AUX_FILE_DIR}`).

### Signal

Triton(2fa1c595f748f239f42e629d96ef48492aedae72) or LLVM(86b69c31642e98f8357df62c09d118ad1da4e16a) has added fma for scalar and vector.