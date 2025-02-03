#include <iostream>
#include <vector>
#include <random>
#include <chrono>

using namespace std;

void matrix_multiply_direct(const vector<double>& A, const vector<double>& B, vector<double>& C, int M, int N, int P) {
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < P; ++j) {
            double sum = 0.0;
            for (int k = 0; k < N; ++k)
                sum += A[i * N + k] * B[k * P + j];
            C[i * P + j] = sum;
        }
}

void transpose_matrix(const vector<double>& B, vector<double>& BT, int N, int P) {
    for (int i = 0; i < P; ++i)
        for (int j = 0; j < N; ++j)
            BT[i * N + j] = B[j * P + i];
}

void matrix_multiply_transpose(const vector<double>& A, const vector<double>& BT, vector<double>& C, int M, int N, int P) {
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < P; ++j) {
            double sum = 0.0;
            for (int k = 0; k < N; ++k)
                sum += A[i * N + k] * BT[j * N + k];
            C[i * P + j] = sum;
        }
}

int main() {
    const int M = 1500, N = 1500, P = 1500;
    vector<double> A(M * N), B(N * P), C_direct(M * P), C_transpose(M * P), BT(P * N);

    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dist(0, 99);
    for (auto& x : A) x = dist(gen);
    for (auto& x : B) x = dist(gen);

    auto start = chrono::high_resolution_clock::now();
    matrix_multiply_direct(A, B, C_direct, M, N, P);
    auto end = chrono::high_resolution_clock::now();
    cout << A << endl;
    cout << A[0] << endl;
    cout << *A[0] << endl;
    cout << "Direct Multiplication Time " << chrono::duration<double>(end - start).count() << " s\n";

    start = chrono::high_resolution_clock::now();
    transpose_matrix(B, BT, N, P);
    end = chrono::high_resolution_clock::now();
    double transpose_time = chrono::duration<double>(end - start).count();

    start = chrono::high_resolution_clock::now();
    matrix_multiply_transpose(A, BT, C_transpose, M, N, P);
    end = chrono::high_resolution_clock::now();
    double multiply_time = chrono::duration<double>(end - start).count();

    cout << "Transpose-Time: " << transpose_time << " s\n";
    cout << "Multiply-Time: " << multiply_time << " s\n";
    cout << "Optimized Total Multiplication Time: " << transpose_time + multiply_time << " s\n";

    cout << "Check Multiplication Result: " << (C_direct == C_transpose ? "identical" : "different") << endl;
    return 0;
}
