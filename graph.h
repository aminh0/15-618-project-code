#pragma once
#include <vector>

struct Graph {
    int N;
    std::vector<std::vector<int>> adj;

    Graph(int n) : N(n), adj(n) {}

    void addEdge(int u, int v) {
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
};