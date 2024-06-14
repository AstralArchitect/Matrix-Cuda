project('Matrix CUDA', 'cuda')

cuda = meson.get_compiler('cuda')

# Define sources
sources = [
    'Code/main.cu',
    'Code/kernel_call.cu',
    'Code/kernel.cu',
    'Code/matrix.cu'
]

# Path to the CUDA toolkit (adjust according to your installation)
cuda_path = 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4'  # Example for Windows

# Include directory
cuda_inc = include_directories(join_paths(cuda_path, 'include'))

# Library directory
cuda_lib = join_paths(cuda_path, 'lib/x64')

# Find the cuBLAS library
cublas_dep = declare_dependency( include_directories: cuda_inc, link_args: ['-L' + cuda_lib, '-lcublas'])

# Create the executable
executable('main', sources, dependencies: [cublas_dep])
