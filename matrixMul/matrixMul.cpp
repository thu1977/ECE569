#include <iostream>
#include <chrono>
#include <vector>

using namespace std;
using namespace std::chrono;

template <typename T>
class Matrix {
private:
    int size;
    vector<vector<T>> buffer;

public:
    Matrix(int n, bool identity = false) : size(n), buffer(n, vector<T>(n, 0)) {
        if (identity) {
            for (int i = 0; i < n; i++) {
                buffer[i][i] = 1;
            }
        }
    }

    Matrix(const Matrix &other) : size(other.size), buffer(other.buffer) {}

    Matrix &operator=(const Matrix &other) {
        if (this != &other) {
            size = other.size;
            buffer = other.buffer;
        }
        return *this;
    }

    void display() const {
        for (const auto &row : buffer) {
            for (const auto &val : row) {
                cout << val << "\t";
            }
            cout << endl;
        }
    }

    void fillSequential() {
        int counter = 0;
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                buffer[i][j] = counter++;
            }
        }
    }

    Matrix transpose() const {
        Matrix transposed(size);
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                transposed.buffer[j][i] = buffer[i][j];
            }
        }
        return transposed;
    }

    Matrix multiply(const Matrix &other) const {
        Matrix result(size);
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                for (int k = 0; k < size; k++) {
                    result.buffer[i][j] += buffer[i][k] * other.buffer[k][j];
                }
            }
        }
        return result;
    }

    Matrix multiplyTransposed(const Matrix &transposed) const {
        Matrix result(size);
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                for (int k = 0; k < size; k++) {
                    result.buffer[i][j] += buffer[i][k] * transposed.buffer[j][k];
                }
            }
        }
        return result;
    }

    Matrix multiplyBlocked(const Matrix &transposed, int blockSize) const {
        Matrix result(size);
        for (int i = 0; i < size; i += blockSize) {
            for (int j = 0; j < size; j += blockSize) {
                for (int i1 = i; i1 < i + blockSize && i1 < size; i1++) {
                    for (int j1 = j; j1 < j + blockSize && j1 < size; j1++) {
                        for (int k = 0; k < size; k++) {
                            result.buffer[i1][j1] += buffer[i1][k] * transposed.buffer[j1][k];
                        }
                    }
                }
            }
        }
        return result;
    }
};

int main(int argc, char *argv[]) {
    int N = atoi(argv[1]);
    int blockSize = atoi(argv[2]);

    Matrix<int> A(N);
    A.fillSequential();

    Matrix<int> B(N, true);
    Matrix<int> B_transposed = B.transpose();

    cout << "Matrix result = matrix1.multiply(matrix2);" << endl;
    auto start1 = high_resolution_clock::now();
    Matrix<int> C1 = A.multiply(B);
    auto stop1 = high_resolution_clock::now();
    cout << "Time for method 1: " << duration_cast<microseconds>(stop1 - start1).count() << " us\n";

    cout << "Matrix result = matrix1.multiplyTransposed(matrix2.transpose());" << endl;
    auto start2 = high_resolution_clock::now();
    Matrix<int> C2 = A.multiplyTransposed(B_transposed);
    auto stop2 = high_resolution_clock::now();
    cout << "Time for method 2: " << duration_cast<microseconds>(stop2 - start2).count() << " us\n";

    cout << "Matrix result = matrix1.multiplyBlocked(matrix2.transpose(), blockSize);" << endl;
    auto start3 = high_resolution_clock::now();
    Matrix<int> C3 = A.multiplyBlocked(B_transposed, blockSize);
    auto stop3 = high_resolution_clock::now();
    cout << "Time for method 3: " << duration_cast<microseconds>(stop3 - start3).count() << " us\n";

    return 0;
}

// Compile with: g++ -o matrixMul matrixMul.cpp -std=c++11 -O3
// Run with: ./matrixMul 2000 200

// Result:
// tianyu@HuMac matrixMul % ./matrixMul 2000 20
// Matrix result = matrix1.multiply(matrix2);
// Time for method 1: 10194236 us
// Matrix result = matrix1.multiplyTransposed(matrix2.transpose());
// Time for method 2: 2201488 us
// Matrix result = matrix1.multiplyBlocked(matrix2.transpose(), blockSize);
// Time for method 3: 2196616 us
