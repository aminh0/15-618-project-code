#include "graph.h"
#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>

Graph generateGraphConnectivity(int N, double c);

int main(int argc, char* argv[]) {
    int N = -1;
    double c = -1.0;
    std::string basename = "graph"; 

    // ---- parse args ----
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-n" && i + 1 < argc) {
            N = std::atoi(argv[++i]);
        }
        else if (arg == "-c" && i + 1 < argc) {
            c = std::atof(argv[++i]);
        }
        else if (arg == "--out" && i + 1 < argc) {
            basename = argv[++i];
        }
        else {
            std::cerr << "Usage: ./graph_gen -n <nodes> -c <conn> --out <basename>\n";
            return 1;
        }
    }

    if (N <= 0 || c < 0.0 || c > 1.0) {
        std::cerr << "Usage: ./graph_gen -n <nodes> -c <conn> --out <basename>\n";
        return 1;
    }

    // 파일 이름 자동 설정
    std::string outFile = basename + ".txt";
    std::string dotFile = basename + ".dot";
    std::string pngFile = basename + ".png";

    // ---- generate ----
    Graph g = generateGraphConnectivity(N, c);

    // ---- save edge list ----
    std::ofstream fout(outFile);
    for (int u = 0; u < g.N; u++) {
        fout << u << ": ";
        for (int v : g.adj[u]) fout << v << " ";
        fout << "\n";
    }
    fout.close();
    std::cout << "Saved edge list to " << outFile << "\n";

    // ---- save DOT file ----
    std::ofstream fdot(dotFile);
    fdot << "graph G {\n";
    for (int u = 0; u < g.N; u++) {
        fdot << "  " << u << ";\n";
    }
    for (int u = 0; u < g.N; u++) {
        for (int v : g.adj[u]) {
            if (u < v)   
                fdot << "  " << u << " -- " << v << ";\n";
        }
    }
    fdot << "}\n";
    fdot.close();
    std::cout << "Saved DOT graph to " << dotFile << "\n";

    // ---- generate PNG automatically ----
    std::string cmd = "dot -Tpng " + dotFile + " -o " + pngFile;
    int ret = system(cmd.c_str());

    if (ret == 0) {
        std::cout << "Generated PNG: " << pngFile << "\n";
    } else {
        std::cerr << "Error: Graphviz 'dot' command failed.\n";
    }

    return 0;
}
