#include "graph.h"
#include <vector>
#include <algorithm>
#include <random>

Graph generateGraphConnectivity(int N, double c) {
    Graph g(N);

    int maxEdges = N * (N - 1) / 2;
    int targetEdges = (int)(c * maxEdges);

    std::vector<std::pair<int,int>> allEdges;
    allEdges.reserve(maxEdges);

    for (int u = 0; u < N; u++) {
        for (int v = u + 1; v < N; v++) {
            allEdges.emplace_back(u, v);
        }
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(allEdges.begin(), allEdges.end(), gen);

    for (int i = 0; i < targetEdges; i++) {
        auto [u, v] = allEdges[i];
        g.addEdge(u, v);
    }

    return g;
}