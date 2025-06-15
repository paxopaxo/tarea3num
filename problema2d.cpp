#include <iostream>
#include <fstream>
#include <vector>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <cmath>

using namespace std;
using namespace Eigen;

typedef Triplet<double> T;


double u_exact(double x, double y) {
    return sin(M_PI * x) * sin(M_PI * y);
}

double f(double x, double y) {
    return 2 * M_PI * M_PI * sin(M_PI * x) * sin(M_PI * y);
}

double b1(double x, double y) { return 0.0; }  
double b2(double x, double y) { return 0.0; }

void write_csv(const vector<double>& x, const vector<double>& y, const VectorXd& U, int N, const string& filename) {
    ofstream file(filename);
    file << "x,y,u\n";
    for (int j = 0; j < N - 1; ++j) {
        for (int i = 0; i < N - 1; ++i) {
            file << x[i + 1] << "," << y[j + 1] << "," << U[j * (N - 1) + i] << "\n";
        }
    }
    file.close();
}

void solve_pde_2D(int N, double nu, double kappa, const string& filename) {
    double h = 1.0 / N;
    int n = (N - 1) * (N - 1);  // Número de incógnitas
    vector<T> coefficients;
    VectorXd F(n);

    // Coordenadas
    vector<double> x(N + 1), y(N + 1);
    for (int i = 0; i <= N; ++i) {
        x[i] = i * h;
        y[i] = i * h;
    }

    auto idx = [N](int i, int j) { return j * (N - 1) + i; };  // mapping (i,j) -> k

    for (int j = 0; j < N - 1; ++j) {
        for (int i = 0; i < N - 1; ++i) {
            int k = idx(i, j);
            double xi = x[i + 1];
            double yj = y[j + 1];

            double b1_ = b1(xi, yj);
            double b2_ = b2(xi, yj);

            double center = 4 * nu / (h * h) + kappa;
            double east   = -nu / (h * h) - b1_ / (2 * h);
            double west   = -nu / (h * h) + b1_ / (2 * h);
            double north  = -nu / (h * h) - b2_ / (2 * h);
            double south  = -nu / (h * h) + b2_ / (2 * h);

            // Centro
            coefficients.emplace_back(k, k, center);

            // Oeste
            if (i > 0)
                coefficients.emplace_back(k, idx(i - 1, j), west);
            else
                F(k) -= west * 0;  // frontera u = 0

            // Este
            if (i < N - 2)
                coefficients.emplace_back(k, idx(i + 1, j), east);
            else
                F(k) -= east * 0;

            // Sur
            if (j > 0)
                coefficients.emplace_back(k, idx(i, j - 1), south);
            else
                F(k) -= south * 0;

            // Norte
            if (j < N - 2)
                coefficients.emplace_back(k, idx(i, j + 1), north);
            else
                F(k) -= north * 0;

            F(k) += f(xi, yj);
        }
    }

    SparseMatrix<double> A(n, n);
    A.setFromTriplets(coefficients.begin(), coefficients.end());

    SparseLU<SparseMatrix<double>> solver;
    solver.compute(A);
    VectorXd U = solver.solve(F);

    write_csv(x, y, U, N, filename);

    // Esto crea el vector de errores que despues se usa en python para calcular la norma 
    VectorXd E(U.size());

    for (int j = 0; j < N - 1; ++j) {
        for (int i = 0; i < N - 1; ++i) {
            int k = j * (N - 1) + i;
            double xi = x[i + 1];
            double yj = y[j + 1];
            E(k) = U(k) - u_exact(xi, yj);
        }
    }

    // for (int j = 0; j < N - 1; ++j) {
    //     for (int i = 0; i < N - 1; ++i) {
    //         int k = j * (N - 1) + i;
    //         double xi = (i + 1) * h;  // en vez de x[i + 1]
    //         double yj = (j + 1) * h;  // en vez de y[j + 1]
    //         E(k) = U(k) - u_exact(xi, yj);
    //     }
    // }


    ofstream file("error_" + filename);
    file << "x,y,e\n";
    for (int j = 0; j < N - 1; ++j) {
        for (int i = 0; i < N - 1; ++i) {
            int k = j * (N - 1) + i;
            file << x[i + 1] << "," << y[j + 1] << "," << E(k) << "\n";
        }
    }
    file.close();
}

int main(int argc, char* argv[]) {
    // Verificar que se pasaron los argumentos necesarios
    if (argc != 6) {
        cerr << "Uso: " << argv[0] << " <N> <nu> <kappa> <refinamiento> <iteraciones>" << endl;
        cerr << "Ejemplo: " << argv[0] << " 16 1.0 0.0 1.0 1.0" << endl;
        return 1;
    }

    // Leer parámetros desde la consola
    int N = atoi(argv[1]);         // Puntos por dirección
    double nu = atof(argv[2]);     // Coeficiente de difusión
    double kappa = atof(argv[3]);  // Coeficiente de reacción
    int refinamiento = atof(argv[4]);     // Componente x convectiva
    int iteraciones = atof(argv[5]);     // Componente y convectiva
    
    // int N = 16;  // puntos por lado (N+1 nodos, N-1 incógnitas por dirección)
    // double nu = 1.0;
    // double kappa = 0.0;
    // int refinamiento = 2;
    // int iteraciones = 5;

    for (int r = 0; r < iteraciones; ++r) {
        string filename = "solucion_N" + to_string(N) + ".csv";
        cout << "Resolviendo con N = " << N << endl;
        solve_pde_2D(N, nu, kappa, filename);
        N *= refinamiento;
    }

    return 0;
}
