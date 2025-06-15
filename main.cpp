#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <fstream>
#include <vector>
#include <cmath>
#include "json/single_include/nlohmann/json.hpp"

using json = nlohmann::json;

void escribirSolucion(const std::vector<double>& x, const Eigen::VectorXd& u, double v) {
    // Calcula el exponente base 10 de v
    int exponente = static_cast<int>(std::round(std::log10(v)));
    // Construye el nombre del archivo como "solucion[exponente].csv"
    std::string nombreArchivo = "solucion" + std::to_string(exponente) + ".csv";
    std::ofstream archivo(nombreArchivo);

    if (!archivo.is_open()) {
        std::cerr << "Error al abrir el archivo para escribir la solución." << std::endl;
        return;
    }

    archivo << "i,x,y\n";
    for (int i = 0; i < x.size(); ++i) {
        archivo << i << "," << x[i] << "," << u[i] << "\n";
        std::cout << "x = " << x[i] << ", u = " << u[i] << '\n'; // tambien lo imprime en pantalla
    }

    archivo.close();
    std::cout << "Solución escrita en solucion.csv" << std::endl;
}

// Es la función f(x) = 1
double localf(double x) {
    return 1.0;
}


int main(int argc, char* argv[]) {
    if (argc != 5) {
        std::cerr << "Modo uso: " << argv[0] << " <puntos_iniciales> <factor_refinamiento> <num_iteraciones> <v param>";
        return 1;
    }

    // Leer parámetros desde la línea de comandos
    int point = std::atoi(argv[1]);      // Puntos iniciales
    int factor = std::atoi(argv[2]);     // Factor de refinamiento
    int nIter = std::atoi(argv[3]);      // Número de refinamientos
    double v = std::stod(argv[4]); 

    std::ifstream archivo("params.json");
    if (!archivo.is_open()) {
        std::cerr << "No se pudo abrir el archivo JSON" << std::endl;
        return 1;
    }

    json j;
    archivo >> j;  // Leer directamente desde el archivo de los parametros
    // Guardar como float
    // double v     = j["v"];
    double kappa = j["kappa"];
    double b_x   = j["b_x"];
    double alpha = j["alpha"];
    double beta = j["beta"];

    // Mostrar los nombres de las variables 
    std::cout << "v: "     << v     << std::endl;
    std::cout << "kappa: " << kappa << std::endl;
    std::cout << "b_x: "   << b_x   << std::endl;
    std::cout << "alpha: " << alpha << std::endl;
    std::cout << "beta: "  << beta  << std::endl;


    std::vector<double> points;     // definir el vector de los puntos ( será por ejemplo (h, 2h, 3h, etc..) )
    Eigen::VectorXd u;              // Vector u en el problema 

    for (int iter = 0; iter < nIter; ++iter) {
        double h = 1.0 /point;
        double h_squared = h * h;
        int N = point - 1;  //  puntos interiores
        
        // Calcular coeficientes
        double a = -v / h_squared - b_x / (2.0 * h);
        double b =  (2.0 * v) / h_squared + kappa;
        double c = -v / h_squared + b_x / (2.0 * h);

        std::cout << "h: "     << h     << std::endl;
        std::cout << "a: "     << a     << std::endl;
        std::cout << "b: " << b << std::endl;
        std::cout << "c: "   << c   << std::endl;



        // Actualizar vector de puntos
        points.resize(N);   // mapea el vector de los puntos de la forma (h, 2h, 3h, etc..)
        for (int i = 0; i < N; ++i) {
            points[i] = (i+1) * h;
        }

        std::cout << "Puntos interiores:\n";
        for (int i = 0; i < points.size(); ++i) {
            std::cout << "points[" << i << "] = " << points[i] << std::endl;
        }


        // Creación matriz A sparse para ahorrar memoria.
        Eigen::SparseMatrix<double> A(N, N);
        
        // Se crea esta matriz ya que por lo visto analiticamente resolver esta matriz resuelve el problema.
        
        for (int i = 0; i < N; ++i) {
            A.coeffRef(i, i) = b; // diagonal
            if (i > 0)
                A.coeffRef(i, i - 1) = a; // subdiagonal
            if (i < N - 1)
                A.coeffRef(i, i + 1) = c; // superdiagonal
        }

        std::cout << "Matriz A:\n" << A << std::endl;



        // Crear vector F
        Eigen::VectorXd F(N);
        
        for (int i = 0; i < N; ++i) {
            F(i) = localf(points[i]);
        }
        F(0) -= alpha*a;
        F(N-1) -= c*beta;

        std::cout << "Vector F:\n" << F << std::endl;

        // Resolver A * u = F
        Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
        solver.analyzePattern(A);
        solver.factorize(A);
        u = solver.solve(F);

        // Refinar para la siguiente iteración
        point *= factor;
    }
    escribirSolucion(points, u, v);

    return 0;
}
