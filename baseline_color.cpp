// g++ -O2 -std=c++17 -fopenmp baseline_color.cpp -o baseline
// ./baseline numnodes prob mode

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

int speculative_parallel_greedy_color(const Graph &g,
                                      std::vector<int> &color,
                                      int num_threads)
{
    int n = g.n;
    color.assign(n, -1);

    // vertex ordering
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);

    omp_set_num_threads(num_threads);

    bool changed = true;
    int iter = 0;

    while (changed) {
        changed = false;
        iter++;

        // ---- Phase 1: tentative parallel greedy coloring ----
#pragma omp parallel for schedule(dynamic, 64)
        for (int idx = 0; idx < n; ++idx) {
            int u = order[idx];

            // Tentative coloring without considering conflicts
            std::vector<char> used(64, 0); // assume small chromatic number
                                           // grow if necessary

            for (int v : g.adj[u]) {
                int c = color[v];
                if (c >= 0) {
                    if (c >= (int)used.size()) used.resize(c + 1, 0);
                    used[c] = 1;
                }
            }

            int c = 0;
            while (c < (int)used.size() && used[c]) c++;
            color[u] = c;
        }

        // ---- Phase 2: detect conflicts ----
        // If two adjacent vertices share the same color,
        // the higher-index vertex will repaint in next iteration.
#pragma omp parallel for schedule(dynamic, 64) reduction(||:changed)
        for (int u = 0; u < n; ++u) {
            for (int v : g.adj[u]) {
                if (u < v && color[u] == color[v]) {
                    color[v] = -1; // mark for recoloring
                    changed = true;
                }
            }
        }
    }

    // compute max color
    int max_color = 0;
    for (int c : color) max_color = std::max(max_color, c);
    return max_color + 1;
}

int luby_parallel_mis_color(const Graph &g,
                            std::vector<int> &color,
                            int num_threads)
{
    int n = g.n;
    color.assign(n, -1);

    omp_set_num_threads(num_threads);

    std::vector<char> active(n, 1);     // still uncolored
    std::vector<float> priority(n);     // random priority values

    std::mt19937 base_rng(12345);
    std::uniform_real_distribution<float> dist(0.0, 1.0);

    int current_color = 0;

    while (true) {
        bool has_active = false;

        // ---- Step 1: assign random priorities to active vertices ----
#pragma omp parallel for reduction(||:has_active)
        for (int u = 0; u < n; ++u) {
            if (active[u]) {
                priority[u] = dist(base_rng);
                has_active = true;
            }
        }

        if (!has_active) break; 

        // ---- Step 2: select local-maxima vertices = MIS ----
        std::vector<char> inMIS(n, 0);

#pragma omp parallel for schedule(dynamic, 64)
        for (int u = 0; u < n; ++u) {
            if (!active[u]) continue;

            float pu = priority[u];
            bool highest = true;

            for (int v : g.adj[u]) {
                if (active[v] && priority[v] > pu) {
                    highest = false;
                    break;
                }
            }
            if (highest) inMIS[u] = 1;
        }

        // ---- Step 3: color MIS and deactivate them ----
#pragma omp parallel for
        for (int u = 0; u < n; ++u) {
            if (inMIS[u]) {
                color[u] = current_color;
                active[u] = 0;
            }
        }

        current_color++;
    }

    return current_color;
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
    int num_threads = 8;
    int mode = 2; // default = Luby MIS

    if (argc > 1) n = std::stoi(argv[1]);
    if (argc > 2) p = std::stod(argv[2]);
    if (argc > 3) mode = std::stoi(argv[3]);
    if (argc > 4) num_iters = std::stoi(argv[4]);
    if (argc > 5) num_threads = std::stoi(argv[5]);

    omp_set_num_threads(num_threads);

    std::cout << "Generating random graph: n=" << n << " p=" << p << "\n";
    Graph g = generate_random_graph(n, p);

    std::vector<int> color;

    double t0 = omp_get_wtime();

    int num_colors = -1;
    if (mode == 0) {
        std::cout << "[Mode 0] Sequential greedy coloring\n";
        num_colors = greedy_color(g, color);
    }
    else if (mode == 1) {
        std::cout << "[Mode 1] Speculative parallel greedy coloring\n";
        num_colors = speculative_parallel_greedy_color(g, color, num_threads);
    }
    else if (mode == 2) {
        std::cout << "[Mode 2] Luby MIS-based parallel coloring\n";
        num_colors = luby_parallel_mis_color(g, color, num_threads);
    }
    else {
        std::cerr << "Invalid mode. Use 0,1,or 2.\n";
        return 1;
    }

    double t1 = omp_get_wtime();

    std::cout << "Coloring used " << num_colors
              << " colors (time: " << (t1 - t0) << " s)\n";

    if(num_colors < 0) {
        std::cerr << "Coloring failed.\n";
        return 1;
    }

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

    // std::cout << "vertex values after smoothing:\n";
    // for (int i = 0; i < std::min(n, 10); ++i) {
    //     std::cout << "v " << i << " = " << values[i] << "\n";
    // }

    return 0;
}