// g++ -O2 -std=c++17 -fopenmp baseline_color.cpp -o baseline
// ./baseline numnodes prob

#include <bits/stdc++.h>
#include <omp.h>

struct Graph {
    int n;
    std::vector<std::vector<int>> adj;

    Graph(int n = 0) : n(n), adj(n) {}

    void add_edge(int u, int v) {
        if (u == v) return;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
};

Graph generate_random_graph(int n, double p, unsigned seed = 42) {
    Graph g(n);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    for (int u = 0; u < n; ++u) {
        for (int v = u + 1; v < n; ++v) {
            if (dist(rng) < p) {
                g.add_edge(u, v);
            }
        }
    }
    return g;
}

int greedy_color(const Graph &g, std::vector<int> &color) {
    int n = g.n;
    color.assign(n, -1);
    std::vector<char> used(n, 0);

    int max_color = -1;
    for (int u = 0; u < n; ++u) {
        for (int v : g.adj[u]) {
            if (color[v] != -1) used[color[v]] = 1;
        }
        int c = 0;
        while (c <= max_color && used[c]) ++c;
        if (c > max_color) max_color = c;
        color[u] = c;
        for (int v : g.adj[u]) {
            if (color[v] != -1) used[color[v]] = 0;
        }
    }
    return max_color + 1; 
}

std::vector<std::vector<int>> build_color_classes(const std::vector<int> &color,
                                                  int num_colors) {
    std::vector<std::vector<int>> classes(num_colors);
    int n = (int)color.size();
    for (int v = 0; v < n; ++v) {
        int c = color[v];
        if (c >= 0 && c < num_colors) {
            classes[c].push_back(v);
        }
    }
    return classes;
}

void laplacian_smoothing_color_batches(
    const Graph &g,
    const std::vector<std::vector<int>> &color_classes,
    std::vector<double> &values,
    int num_iters)
{
    int n = g.n;
    std::vector<double> tmp(n);

    for (int iter = 0; iter < num_iters; ++iter) {
        tmp = values;

        for (size_t c = 0; c < color_classes.size(); ++c) {
            const auto &cls = color_classes[c];

#pragma omp parallel for schedule(dynamic, 64)
            for (size_t idx = 0; idx < cls.size(); ++idx) {
                int v = cls[idx];
                const auto &nbrs = g.adj[v];

                if (nbrs.empty()) {
                    tmp[v] = values[v];
                    continue;
                }

                double sum = 0.0;
                for (int u : nbrs) {
                    sum += values[u];
                }
                tmp[v] = sum / (double)nbrs.size();
            }
        }

        values.swap(tmp);
    }
}

int main(int argc, char **argv) {
    int n = 200;
    double p = 0.05;
    int num_iters = 10;
    int num_threads = 4;

    if (argc > 1) n = std::stoi(argv[1]);
    if (argc > 2) p = std::stod(argv[2]);
    if (argc > 3) num_iters = std::stoi(argv[3]);
    if (argc > 4) num_threads = std::stoi(argv[4]);

    omp_set_num_threads(num_threads);

    std::cout << "Generating random graph: n=" << n << " p=" << p << "\n";
    Graph g = generate_random_graph(n, p);

    std::vector<int> color;
    double t0 = omp_get_wtime();
    int num_colors = greedy_color(g, color);
    double t1 = omp_get_wtime();

    std::cout << "Greedy coloring used " << num_colors << " colors (time: " << (t1 - t0) << " s)\n";

    auto color_classes = build_color_classes(color, num_colors);

    std::vector<double> values(n);
    std::mt19937 rng(123);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for (int i = 0; i < n; ++i) values[i] = dist(rng);

    double t2 = omp_get_wtime();
    laplacian_smoothing_color_batches(g, color_classes, values, num_iters);
    double t3 = omp_get_wtime();

    std::cout << "Laplacian smoothing (" << num_iters << " iters) runtime: "
              << (t3 - t2) << " s with " << num_threads << " threads\n";

    std::cout << "vertex values after smoothing:\n";
    for (int i = 0; i < std::min(n, 10); ++i) {
        std::cout << "v " << i << " = " << values[i] << "\n";
    }

    return 0;
}
