

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cassert>
#include <cmath>
#include <ctime>
#include <cuda_runtime.h>
#include <cublas_v2.h>

using namespace std;

void checkCuda(cudaError_t result) {
    if (result != cudaSuccess) {
        cerr << "CUDA Runtime Error: " << cudaGetErrorString(result) << endl;
        assert(result == cudaSuccess);
    }
}

void checkCuBLAS(cublasStatus_t result) {
    if (result != CUBLAS_STATUS_SUCCESS) {
        cerr << "cuBLAS Library Error" << endl;
        assert(result == CUBLAS_STATUS_SUCCESS);
    }
}

void matVecMultiply(cublasHandle_t handle, double* d_A, double* d_x, double* d_y, int N) {
    const double alpha = 1.0;
    const double beta = 0.0;
    checkCuBLAS(cublasDgemv(handle, CUBLAS_OP_N, N, N, &alpha, d_A, N, d_x, 1, &beta, d_y, 1));
}

vector<double> BiCG_cuBLAS(double* d_A, double* d_b, int N, int max_iter, double tol) {
    cublasHandle_t handle;
    cublasCreate(&handle);

    double* d_x, * d_r, * d_p, * d_Ap;
    double alpha, beta, rdot, r1dot, pap;
    cudaMalloc((void**)&d_x, N * sizeof(double));
    cudaMalloc((void**)&d_r, N * sizeof(double));
    cudaMalloc((void**)&d_p, N * sizeof(double));
    cudaMalloc((void**)&d_Ap, N * sizeof(double));

    cudaMemset(d_x, 0, N * sizeof(double));
    cudaMemcpy(d_r, d_b, N * sizeof(double), cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_p, d_b, N * sizeof(double), cudaMemcpyDeviceToDevice);

    checkCuBLAS(cublasDdot(handle, N, d_r, 1, d_r, 1, &rdot));

    int iter = 0;
    while (iter < max_iter && rdot > tol * tol) {
        matVecMultiply(handle, d_A, d_p, d_Ap, N);
        checkCuBLAS(cublasDdot(handle, N, d_p, 1, d_Ap, 1, &pap));

        alpha = rdot / pap;

        checkCuBLAS(cublasDaxpy(handle, N, &alpha, d_p, 1, d_x, 1));
        alpha = -alpha;
        checkCuBLAS(cublasDaxpy(handle, N, &alpha, d_Ap, 1, d_r, 1));

        r1dot = rdot;
        checkCuBLAS(cublasDdot(handle, N, d_r, 1, d_r, 1, &rdot));

        beta = rdot / r1dot;

        checkCuBLAS(cublasDscal(handle, N, &beta, d_p, 1));
        beta = 1.0;
        checkCuBLAS(cublasDaxpy(handle, N, &beta, d_r, 1, d_p, 1));

        iter++;
    }

    vector<double> x(N);
    cudaMemcpy(x.data(), d_x, N * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_r);
    cudaFree(d_p);
    cudaFree(d_Ap);
    cublasDestroy(handle);

    return x;
}

int main() {
    ofstream g("result2.txt");
    double startAlg = clock();

    vector<string> types = { "good" };
    vector<int> sizes = { 1000 };
    int number = 1;

    for (auto& type : types) {
        for (auto& size : sizes) {
            for (int count = 1; count <= 1; ++count) {
                string fileName = "matrix/" + type + "_cond_" + to_string(size) + "_" + to_string(count) + ".txt";
                ifstream f(fileName);
                vector<double> h_A(size * size), h_b(size);

                for (int i = 0; i < size; i++) {
                    for (int j = 0; j < size; j++) {
                        f >> h_A[i * size + j];
                    }
                    f >> h_b[i];
                }
                f.close();

                double* d_A, * d_b;
                cudaMalloc((void**)&d_A, size * size * sizeof(double));
                cudaMalloc((void**)&d_b, size * sizeof(double));

                cudaMemcpy(d_A, h_A.data(), size * size * sizeof(double), cudaMemcpyHostToDevice);
                cudaMemcpy(d_b, h_b.data(), size * sizeof(double), cudaMemcpyHostToDevice);

                g << endl << number << " " << fileName << endl;
                ++number;
                double start = clock();

                vector<double> b = BiCG_cuBLAS(d_A, d_b, size, 1000000, 0.0000000001);

                double finish = clock();
                for (double i : b) {
                    g << i << "\t";
                }
                g << "\n\n";

                vector<double> res(size, 0.0);
                for (int i = 0; i < size; ++i) {
                    for (int j = 0; j < size; ++j) {
                        res[i] += h_A[i * size + j] * b[j];
                    }
                }

                vector<double> error(size);
                double mx = 0;
                g << "\nErrors:\n";
                for (int i = 0; i < size; ++i) {
                    error[i] = abs(res[i] - h_b[i]);
                    mx = max(mx, error[i]);
                    g << error[i] << "\t";
                }
                g << endl << "Max error: " << mx << endl;
                g << "Time: " << (finish - start) / CLOCKS_PER_SEC << " сек\n";

                cudaFree(d_A);
                cudaFree(d_b);
            }
        }
    }

    g.close();
    double endAlg = clock();
    cout << "Total time: " << (endAlg - startAlg) / CLOCKS_PER_SEC << " sec" << endl;

    return 0;
}






