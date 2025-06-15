
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <string>
#include <Eigen/Sparse>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

typedef Triplet<double> T;

// Parámetros físicos del problema
const double nu = 1.0;       // coeficiente difusivo
const double kappa = 0.0;    // coeficiente de reacción
const double pi = M_PI;

// Campos advectivos sin divergencia
double b1(double x, double y) { return 0.0; }
double b2(double x, double y) { return 0.0; }

// Solución exacta 
double u_exact(double x, double y) {
    return sin(pi * x) * sin(pi * y);
}

// RHS calculado a partir de la solución exacta
double f_rhs(double x, double y) {
    double uxx = -pi * pi * sin(pi * x) * sin(pi * y);
    double uyy = -pi * pi * sin(pi * x) * sin(pi * y);
    double ux  = pi * cos(pi * x) * sin(pi * y);
    double uy  = pi * sin(pi * x) * cos(pi * y);
    return -nu * (uxx + uyy) + b1(x, y) * ux + b2(x, y) * uy + kappa * u_exact(x, y);
}

// Función de frontera Dirichlet
double u_dirichlet(double x, double y) {
    return u_exact(x, y);
}

// Indexación en 1D del nodo (i, j)
int idx(int i, int j, int N) {
    return (j - 1) * (N - 1) + (i - 1);
}

// Ensambla el sistema lineal AU = F
void build_system(int N, SparseMatrix<double>& A, VectorXd& F, double h) {
    int n = (N - 1) * (N - 1);
    vector<T> coefficients;
    F = VectorXd::Zero(n);

    for (int j = 1; j < N; ++j) {
        for (int i = 1; i < N; ++i) {
            int k = idx(i, j, N);
            double x = i * h;
            double y = j * h;

            double bij1 = b1(x, y);
            double bij2 = b2(x, y);

            double a_c  = 4 * nu / (h * h) + kappa;
            double a_e  = - (nu / (h * h) - bij1 / (2 * h));
            double a_w  = - (nu / (h * h) + bij1 / (2 * h));
            double a_n  = - (nu / (h * h) - bij2 / (2 * h));
            double a_s  = - (nu / (h * h) + bij2 / (2 * h));

            coefficients.push_back(T(k, k, a_c));
            if (i + 1 < N) coefficients.push_back(T(k, idx(i + 1, j, N), a_e));
            else F(k) -= a_e * u_dirichlet(1.0, y);
            if (i - 1 > 0) coefficients.push_back(T(k, idx(i - 1, j, N), a_w));
            else F(k) -= a_w * u_dirichlet(0.0, y);
            if (j + 1 < N) coefficients.push_back(T(k, idx(i, j + 1, N), a_n));
            else F(k) -= a_n * u_dirichlet(x, 1.0);
            if (j - 1 > 0) coefficients.push_back(T(k, idx(i, j - 1, N), a_s));
            else F(k) -= a_s * u_dirichlet(x, 0.0);

            F(k) += f_rhs(x, y);
        }
    }
    A.setFromTriplets(coefficients.begin(), coefficients.end());
}

// Exporta solución o errores
void export_csv(const string& filename, const VectorXd& U, int N, double h, bool is_error = false) {
    ofstream file(filename);
    file << "x,y," << (is_error ? "error\n" : "u_h\n");
    for (int j = 1; j < N; ++j) {
        for (int i = 1; i < N; ++i) {
            int k = idx(i, j, N);
            double x = i * h;
            double y = j * h;
            double val = is_error ? fabs(U(k) - u_exact(x, y)) : U(k);
            file << x << "," << y << "," << val << "\n";
        }
    }
    file.close();
}

int main() {
    int N0 = 33; // puntos interiores + 1
    int refinamientos = 8;

    ofstream summary("summary.csv");
    summary << "N,L2h,Linf\n";

    for (int r = 0; r < refinamientos; ++r) {
        int N = (1 << r) * (N0 - 1) + 1;
        double h = 1.0 / N;
        int n = (N - 1) * (N - 1);

        SparseMatrix<double> A(n, n);
        VectorXd F(n), U(n);

        build_system(N, A, F, h);
        SparseLU<SparseMatrix<double>> solver;
        solver.analyzePattern(A);
        solver.factorize(A);
        U = solver.solve(F);

        export_csv("solucion_N" + to_string(N) + ".csv", U, N, h, false);
        export_csv("error_solucion_N" + to_string(N) + ".csv", U, N, h, true);

        cout << "Refinamiento N=" << N << " completado.\n";
    }

    summary.close();
    return 0;
}
