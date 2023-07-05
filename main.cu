#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

//======================
#define DEV_NO 0
#define CUDA_CHECK_RETURN(value) { \
    cudaError_t _m_cudaStat = value; \
    if (_m_cudaStat != cudaSuccess) { \
        fprintf(stderr, "Error %s at line %d in file %s\n", cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__); \
        exit(1); \
    }}
#define B 32

const int INF = ((1 << 30) - 1);

__constant__ int d_n;
int n, m;
int padded_n;

int* Dist;
void input(char* infile);
void output(char* outFileName);

inline int ceil(int a, int b) { return (a + b - 1) / b; }

__global__ void phaseOne(int* d_dist, int r) {
    int row = r * B + threadIdx.y;
    int col = r * B + threadIdx.x;

    __shared__ int cache[B][B];
    cache[threadIdx.y][threadIdx.x] = d_dist[row * d_n + col];
    __syncthreads();

    for (int i = 0; i < B; ++i) {
        int distance = cache[threadIdx.y][i] + cache[i][threadIdx.x];
        cache[threadIdx.y][threadIdx.x] = min(cache[threadIdx.y][threadIdx.x], distance);
        __syncthreads();
    }

    d_dist[row * d_n + col] = cache[threadIdx.y][threadIdx.x];
}

__global__ void phaseTwoH(int* d_dist, int r) {
    int row = r * B + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ int cache[B][B];
    cache[threadIdx.y][threadIdx.x] = d_dist[row * d_n + col];
    __syncthreads();

    for (int k = r * B; k < (r + 1) * B; ++k) {
        int distance = d_dist[row * d_n + k] + d_dist[k * d_n + col];
        cache[threadIdx.y][threadIdx.x] = min(cache[threadIdx.y][threadIdx.x], distance);
    }

    d_dist[row * d_n + col] = cache[threadIdx.y][threadIdx.x];
}

__global__ void phaseTwoV(int* d_dist, int r) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = r * B + threadIdx.x;

    __shared__ int cache[B][B];
    cache[threadIdx.y][threadIdx.x] = d_dist[row * d_n + col];
    __syncthreads();

    for (int k = r * B; k < (r + 1) * B; ++k) {
        int distance = d_dist[row * d_n + k] + d_dist[k * d_n + col];
        cache[threadIdx.y][threadIdx.x] = min(cache[threadIdx.y][threadIdx.x], distance);
    }

    d_dist[row * d_n + col] = cache[threadIdx.y][threadIdx.x];
}

__global__ void phaseTri(int* d_dist, int r) {
    int row = 2 * (blockIdx.y * blockDim.y + threadIdx.y);
    int col = 2 * (blockIdx.x * blockDim.x + threadIdx.x);

    int y = 2 * threadIdx.y;
    int x = 2 * threadIdx.x;

    __shared__ int cache[B][B];
    __shared__ int cache_row_k[B][B];
    __shared__ int cache_k_col[B][B];

    for (int j = 0; j < 2; ++j) {
        for (int i = 0; i < 2; ++i) {
            cache[y + j][x + i] = d_dist[(row + j) * d_n + (col + i)];
            cache_row_k[y + j][x + i] = d_dist[(row + j) * d_n + ((x + i) + r * B)];
            cache_k_col[y + j][x + i] = d_dist[((y + j) + r * B) * d_n + (col + i)];
        }
    }
    __syncthreads();

    for (int j = 0; j < 2; ++j) {
        for (int i = 0; i < 2; ++i) {
            for (int k = 0; k < B; ++k) {
                int distance = cache_row_k[y + j][k] + cache_k_col[k][x + i];
                cache[y + j][x + i] = min(cache[y + j][x + i], distance);
            }
        }
    }

    for (int j = 0; j < 2; ++j) {
        for (int i = 0; i < 2; ++i) {
            d_dist[(row + j) * d_n + (col + i)] = cache[y + j][x + i];
        }
    }
}

int main(int argc, char* argv[]) {
    input(argv[1]);

    int* d_dist;

    CUDA_CHECK_RETURN(cudaMalloc((void**) &d_dist, padded_n * padded_n * sizeof(int)));
    CUDA_CHECK_RETURN(cudaMemcpy(d_dist, Dist, padded_n * padded_n * sizeof(int), cudaMemcpyHostToDevice));
    
    int round = ceil(n, B);

    dim3 grid(round, round);
    dim3 horizontal(round, 1);
    dim3 vertical(1, round);
    dim3 threadsPerBlock(B, B);
    dim3 phaseTriThreadsPerblock(B/2, B/2);

    cudaStream_t horizontalStream, verticalStream;
    cudaStreamCreate(&horizontalStream);
    cudaStreamCreate(&verticalStream);

    // cudaStream_t upperLeftStream, upperRightStream, downLeftStream, downRightStream;
    // cudaStreamCreate(&upperLeftStream);
    // cudaStreamCreate(&upperRightStream);
    // cudaStreamCreate(&downLeftStream);
    // cudaStreamCreate(&downRightStream);

    for (int r = 0; r < round; ++r) {
        phaseOne <<<1, threadsPerBlock>>>(d_dist, r);
        phaseTwoH <<<horizontal, threadsPerBlock, 0, horizontalStream>>>(d_dist, r);
        phaseTwoV <<<vertical, threadsPerBlock, 0, verticalStream>>>(d_dist, r);
        // dim3 upper_left(r, r);
        // dim3 upper_right(r, round - r - 1);
        // dim3 down_left(round - r - 1, r);
        // dim3 down_right(round - r - 1, round - r - 1);
        // cudaDeviceSynchronize();
        // phaseTriUL <<<upper_left, threadsPerBlock>>>(d_dist, r);
        // phaseTriUR <<<upper_right, threadsPerBlock>>>(d_dist, r);
        // phaseTriDL <<<down_left, threadsPerBlock>>>(d_dist, r);
        // phaseTriDR <<<down_right, threadsPerBlock>>>(d_dist, r);
        phaseTri <<<grid, phaseTriThreadsPerblock>>>(d_dist, r);
    }

    CUDA_CHECK_RETURN(cudaDeviceSynchronize());

    CUDA_CHECK_RETURN(cudaMemcpy(Dist, d_dist, padded_n * padded_n * sizeof(int), cudaMemcpyDeviceToHost));
    
    output(argv[2]);
    
    CUDA_CHECK_RETURN(cudaFree(d_dist));

    return 0;
}

void input(char* infile) {
    FILE* file = fopen(infile, "rb");
    fread(&n, sizeof(int), 1, file);

    int remainder = n % B;

    if (remainder == 0) {
        padded_n = n;
    } else {
        padded_n = n + (B - remainder);
    }

    Dist = new int[padded_n * padded_n];

    cudaMemcpyToSymbol(d_n, &padded_n, sizeof(int));
    fread(&m, sizeof(int), 1, file);

    for (int i = 0; i < padded_n; ++i) {
        for (int j = 0; j < padded_n; ++j) {
            if (i == j) {
                Dist[i * padded_n + j] = 0;
            } else {
                Dist[i * padded_n + j] = INF;
            }
        }
    }

    int pair[3];
    for (int i = 0; i < m; ++i) {
        fread(pair, sizeof(int), 3, file);
        Dist[pair[0] * padded_n + pair[1]] = pair[2];
    }
    fclose(file);
}

void output(char* outFileName) {
    FILE* outfile = fopen(outFileName, "w");
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (Dist[i * padded_n + j] >= INF) Dist[i * padded_n + j] = INF;
        }
        fwrite(&Dist[i*padded_n], sizeof(int), n, outfile);
    }
    fclose(outfile);

    delete[] Dist;
}

