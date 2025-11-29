#include <bits/stdc++.h>
#include <omp.h>
// g++ -O2 -std=c++17 -fopenmp baseline_color.cpp -o baseline
//   ./baseline -n 10000 -p 0.5 -t 8 -i 20 -b 
//   ./baseline -n 10000 -p 0.5 -t 8 -i 20 -l 
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

int conflicts_if(const Graph &g, const std::vector<int> &color, int v, int c) {
    int cnt = 0;
    for (int u : g.adj[v]) {
        if (color[u] == c) ++cnt;
    }
    return cnt;
}

long long total_conflicts(const Graph &g, const std::vector<int> &color) {
    long long conflicts = 0;
    for (int u = 0; u < g.n; ++u) {
        for (int v : g.adj[u]) {
            if (u < v && color[u] == color[v]) ++conflicts;
        }
    }
    return conflicts;
}

void burke_refine(const Graph &g, std::vector<int> &color, int num_colors, int max_iters)
{
    int n = g.n;
    std::vector<int> new_color(n);

    for (int iter = 0; iter < max_iters; ++iter) {
        int changes = 0;

        #pragma omp parallel for schedule(dynamic, 64) reduction(+:changes)
        for (int v = 0; v < n; ++v) {
            int cur_c = color[v];
            int best_c = cur_c;
            int best_conf = conflicts_if(g, color, v, cur_c);

            for (int c = 0; c < num_colors; ++c) {
                if (c == cur_c) continue;
                int c_conf = conflicts_if(g, color, v, c);
                if (c_conf < best_conf || (c_conf == best_conf && c < best_c)) {
                    best_conf = c_conf;
                    best_c = c;
                }
            }

            new_color[v] = best_c;
            if (best_c != cur_c) ++changes;
        }

        color.swap(new_color);

        if (changes == 0) {
            break;
        }
    }
}

std::vector<std::vector<int>> build_color_classes(const std::vector<int> &color, int num_colors) {
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

void laplacian_smoothing_color_batches( const Graph &g, const std::vector<std::vector<int>> &color_classes, std::vector<double> &values, int num_iters)
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

enum Mode {
    MODE_NONE = 0,
    MODE_GREEDY,
    MODE_BURKE,
    MODE_LAPLACIAN
};

int main(int argc, char **argv) {
    int n = -1;
    double p = -1.0;
    int iters = 10;
    int num_threads = 4;
    Mode mode = MODE_NONE;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-n") {
            if (i + 1 >= argc) {
                std::cerr << "Error: -n requires a value\n";
                return 1;
            }
            n = std::atoi(argv[++i]);
        } else if (arg == "-p") {
            if (i + 1 >= argc) {
                std::cerr << "Error: -p requires a value\n";
                return 1;
            }
            p = std::atof(argv[++i]);
        } else if (arg == "-t") {
            if (i + 1 >= argc) {
                std::cerr << "Error: -t requires a value\n";
                return 1;
            }
            num_threads = std::atoi(argv[++i]);
        } else if (arg == "-i") {
            if (i + 1 >= argc) {
                std::cerr << "Error: -i requires a value\n";
                return 1;
            }
            iters = std::atoi(argv[++i]);
        } else if (arg == "-g") {
            if (mode != MODE_NONE) {
                std::cerr << "Error: specify only of -b, or -l.\n";
                return 1;
            }
            mode = MODE_GREEDY;
        } else if (arg == "-b") {
            if (mode != MODE_NONE) {
                std::cerr << "Error: specify only one of -b, or -l.\n";
                return 1;
            }
            mode = MODE_BURKE;
        } else if (arg == "-l") {
            if (mode != MODE_NONE) {
                std::cerr << "Error: specify only one of -b, or -l.\n";
                return 1;
            }
            mode = MODE_LAPLACIAN;
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            return 1;
        }
    }

    if (n <= 0 || p < 0.0 || p > 1.0 || mode == MODE_NONE) {
        return 1;
    }

    omp_set_num_threads(num_threads);

    std::cout << "Generating random graph: n=" << n << " p=" << p << " threads=" << num_threads << "\n";
    Graph g = generate_random_graph(n, p);

    std::vector<int> color;

    double t0 = omp_get_wtime();
    int num_colors = greedy_color(g, color);
    double t1 = omp_get_wtime();
    double t_greedy = t1 - t0;
    long long conf_after_greedy = total_conflicts(g, color);

    std::cout << "Greedy: colors=" << num_colors << "\n";

    if (mode == MODE_BURKE) {
        double t2 = omp_get_wtime();
        burke_refine(g, color, num_colors, iters);
        double t3 = omp_get_wtime();
        double t_burke = t3 - t2;
        long long conf_after_burke = total_conflicts(g, color);

        std::cout << "Burke: time=" << t_burke
                  << " s, conflicts=" << conf_after_burke << "\n";

        return 0;
    }

    if (mode == MODE_LAPLACIAN) {
        auto color_classes = build_color_classes(color, num_colors);

        int n_local = g.n;
        std::vector<double> values(n_local);
        std::mt19937 rng(123);
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        for (int i = 0; i < n_local; ++i) values[i] = dist(rng);

        double t4 = omp_get_wtime();
        laplacian_smoothing_color_batches(g, color_classes, values, iters);
        double t5 = omp_get_wtime();
        double t_lap = t5 - t4;

        std::cout << "Laplacian: time=" << t_lap << " s\n";
        return 0;
    }

    return 0;
}