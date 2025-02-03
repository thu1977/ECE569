#include <iostream>
#include <vector>
#include <chrono>
#include <random>

using namespace std;

// Generate a random matrix of size rows x cols
vector<vector<double> > generateMatrix(size_t rows, size_t cols) {
    vector<vector<double> > matrix(rows, vector<double>(cols));
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(0, 100);

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            matrix[i][j] = dis(gen);
        }
    }
    // //Print out first 5*5 of the matrix
    // for (size_t i = 0; i < 5; ++i) {
    //     for (size_t j = 0; j < 5; ++j) {
    //         cout << matrix[i][j] << " ";
    //     }
    //     cout << endl;
    // }
    return matrix;
}

// Direct multiplication of matrices
vector<vector<double> > directMultiply(const vector<vector<double> >& A, const vector<vector<double> >& B) {
    size_t n = A.size();
    size_t m = B.size();
    size_t p = B[0].size();
    vector<vector<double> > C(n, vector<double>(p, 0));

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < p; ++j) {
            for (size_t k = 0; k < m; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return C;
}

// Transpose of a matrix
vector<vector<double> > transpose(const vector<vector<double> >& matrix) {
    size_t rows = matrix.size();
    size_t cols = matrix[0].size();
    vector<vector<double> > transposed(cols, vector<double>(rows));

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            transposed[j][i] = matrix[i][j];
        }
    }
    return transposed;
}

// Multiplication of matrices with transposition optimization
vector<vector<double> > transposeMultiply(const vector<vector<double> >& A, const vector<vector<double> >& B) {
    size_t n = A.size();
    size_t m = B.size();
    size_t p = B[0].size();
    vector<vector<double> > C(n, vector<double>(p, 0));

    vector<vector<double> > B_T = transpose(B);

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < p; ++j) {
            for (size_t k = 0; k < m; ++k) {
                C[i][j] += A[i][k] * B_T[j][k];
            }
        }
    }
    return C;
}

int main() {
    const size_t n = 1500; // Number of rows in A
    const size_t m = 1500; // Number of columns in A and rows in B
    const size_t p = 1500; // Number of columns in B

    // Generate random matrices A and B
    vector<vector<double> > A = generateMatrix(n, m);
    vector<vector<double> > B = generateMatrix(m, p);

    // Direct multiplication
    chrono::high_resolution_clock::time_point start1 = chrono::high_resolution_clock::now();
    vector<vector<double> > C1 = directMultiply(A, B);
    chrono::high_resolution_clock::time_point end1 = chrono::high_resolution_clock::now();
    chrono::duration<double> duration1 = chrono::duration_cast<chrono::duration<double> >(end1 - start1);

    // Transpose optimization multiplication
    chrono::high_resolution_clock::time_point start2 = chrono::high_resolution_clock::now();
    vector<vector<double> > C2 = transposeMultiply(A, B);
    chrono::high_resolution_clock::time_point end2 = chrono::high_resolution_clock::now();
    chrono::duration<double> duration2 = chrono::duration_cast<chrono::duration<double> >(end2 - start2);

    // Print execution times
    cout << "Direct multiplication time: " << duration1.count() << " seconds" << endl;
    cout << "Transposition optimization time: " << duration2.count() << " seconds" << endl;

    return 0;
}
