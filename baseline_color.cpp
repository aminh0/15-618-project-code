#include <bits/stdc++.h>
#include <omp.h>
// g++ -O2 -std=c++17 -fopenmp baseline_color.cpp -o baseline
// ./baseline -n 10000 -p 0.5 -t 8 -i 20 -g -b -c 20

//   -g   : sequential greedy_color
//   -gp   : speculative_parallel_greedy_color
//   -lb   : luby_parallel_mis_color
//
//   -b   : Burke refinement kernel
//   -l   : Laplacian smoothing kernel
//   -c   :  max color

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

//memory access count
struct Metrics {
    long long color_edge_visits = 0;      // edges traversed during coloring
    long long color_neigh_dist_sum = 0;   // sum |u - v| during coloring

    long long lap_edge_visits = 0;        // edges traversed during Laplacian
    long long lap_neigh_dist_sum = 0;     // sum |u - v| during Laplacian

    double color_locality_score = 0.0;    // computed after coloring (per color class)
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

// ----------------- Sequential greedy coloring -----------------
int greedy_color(const Graph &g, std::vector<int> &color, Metrics &metrics) {
    int n = g.n;
    color.assign(n, -1);
    std::vector<char> used(n, 0);

    int max_color = -1;
    for (int u = 0; u < n; ++u) {
        for (int v : g.adj[u]) {
            metrics.color_edge_visits++;
            metrics.color_neigh_dist_sum += std::llabs((long long)u - (long long)v);
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

// ----------------- Speculative parallel greedy coloring -----------------
int speculative_parallel_greedy_color(const Graph &g, std::vector<int> &color, int num_threads, Metrics &metrics)
{
    int n = g.n;
    color.assign(n, -1);

    // vertex ordering
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);

    omp_set_num_threads(num_threads);

    bool changed = true;

    while (changed) {
        changed = false;

        // ---- Phase 1: tentative parallel greedy coloring ----
        #pragma omp parallel for schedule(dynamic, 64)
        for (int idx = 0; idx < n; ++idx) {
            int u = order[idx];

            // Tentative coloring without considering conflicts
            std::vector<char> used(64, 0); // assume small chromatic number

            for (int v : g.adj[u]) {
                metrics.color_edge_visits++;
                metrics.color_neigh_dist_sum += std::llabs((long long)u - (long long)v);
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

        // ---- Phase 2: detect conflicts and mark for recolor ----
        std::vector<char> mark_recolor(n, 0);

        #pragma omp parallel for schedule(dynamic, 64)
        for (int u = 0; u < n; ++u) {
            for (int v : g.adj[u]) {
                if (u < v && color[u] == color[v]) {
                    // break tie by index: recolor higher index
                    mark_recolor[v] = 1;
                }
            }
        }

        #pragma omp parallel for reduction(||:changed)
        for (int v = 0; v < n; ++v) {
            if (mark_recolor[v]) {
                color[v] = -1;
                changed = true;
            }
        }
    }

    // compute max color
    int max_color = 0;
    for (int c : color) max_color = std::max(max_color, c);
    return max_color + 1;
}

// ----------------- Luby MIS-lbased parallel coloring -----------------
int luby_parallel_mis_color(const Graph &g,
                            std::vector<int> &color,
                            int num_threads)
{
    int n = g.n;
    color.assign(n, -1);

    omp_set_num_threads(num_threads);

    std::vector<char> active(n, 1);     // still uncolored
    std::vector<float> priority(n);     // random priority values

    int current_color = 0;
    const unsigned base_seed = 12345;

    while (true) {
        bool has_active = false;

        // ---- Step 1: assign random priorities to active vertices ----
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            std::mt19937 rng(base_seed + tid * 1337u);
            std::uniform_real_distribution<float> dist(0.0f, 1.0f);

            #pragma omp for reduction(||:has_active)
            for (int u = 0; u < n; ++u) {
                if (active[u]) {
                    priority[u] = dist(rng);
                    has_active = true;
                }
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

// ----------------- Burke refinement -----------------
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

// ----------------- Laplacian smoothing -----------------
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

double compute_color_class_locality(const std::vector<int> &color, int num_colors) {
    if (num_colors <= 0) return 0.0;

    std::vector<std::vector<int>> classes(num_colors);
    int n = (int)color.size();
    for (int v = 0; v < n; ++v) {
        int c = color[v];
        if (c >= 0 && c < num_colors) {
            classes[c].push_back(v);
        }
    }

    double total_std = 0.0;
    int nonempty = 0;

    for (const auto &cls : classes) {
        if (cls.size() <= 1) continue;  // 0 or 1 vertex => stddev = 0, skip

        nonempty++;

        double mean = 0.0;
        for (int v : cls) mean += v;
        mean /= (double)cls.size();

        double var = 0.0;
        for (int v : cls) {
            double d = (double)v - mean;
            var += d * d;
        }
        var /= (double)cls.size();

        total_std += std::sqrt(var);
    }

    if (nonempty == 0) return 0.0;
    return total_std / (double)nonempty;
}


void laplacian_smoothing_color_batches(const Graph &g, const std::vector<std::vector<int>> &color_classes, std::vector<double> &values, int num_iters, Metrics &metrics)
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
                    metrics.lap_edge_visits++;
                    metrics.lap_neigh_dist_sum += std::llabs((long long)u - (long long)v);
                    sum += values[u];
                }
                tmp[v] = sum / (double)nbrs.size();
            }
        }

        values.swap(tmp);
    }
}

// ----------------- Modes & main -----------------
enum ColorMode {
    COLOR_NONE = 0,
    COLOR_GREEDY,
    COLOR_SPECULATIVE,
    COLOR_LUBY
};

enum KernelMode {
    KERNEL_NONE = 0,
    KERNEL_BURKE,
    KERNEL_LAPLACIAN
};

int main(int argc, char **argv) {
    int n = -1;
    double p = -1.0;
    int iters = 10;
    int num_threads = 4;
    ColorMode color_mode = COLOR_NONE;
    KernelMode kernel_mode = KERNEL_NONE;
    int max_colors = -1; // -1 = no limit

    Metrics metrics;

    // ---- parse args ----
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-n") {
            if (i + 1 >= argc) { std::cerr << "Error: -n needs value\n"; return 1; }
            n = std::atoi(argv[++i]);
        } else if (arg == "-p") {
            if (i + 1 >= argc) { std::cerr << "Error: -p needs value\n"; return 1; }
            p = std::atof(argv[++i]);
        } else if (arg == "-t") {
            if (i + 1 >= argc) { std::cerr << "Error: -t needs value\n"; return 1; }
            num_threads = std::atoi(argv[++i]);
        } else if (arg == "-i") {
            if (i + 1 >= argc) { std::cerr << "Error: -i needs value\n"; return 1; }
            iters = std::atoi(argv[++i]);
        } else if (arg == "-g") {
            if (color_mode != COLOR_NONE) {
                std::cerr << "Error: choose only one of -g, -gp, -lb for coloring\n";
                return 1;
            }
            color_mode = COLOR_GREEDY;
        } else if (arg == "-gp") {
            if (color_mode != COLOR_NONE) {
                std::cerr << "Error: choose only one of -g, -gp, -lb for coloring\n";
                return 1;
            }
            color_mode = COLOR_SPECULATIVE;
        } else if (arg == "-c") {
        if (i + 1 >= argc) {
            std::cerr << "Error: -c needs value\n";
            return 1;
        }
        max_colors = std::atoi(argv[++i]);
        if (max_colors <= 0) {
            std::cerr << "Error: -c must be positive\n";
            return 1;
        }
        } else if (arg == "-lb") {       // Luby MIS coloring
            if (color_mode != COLOR_NONE) {
                std::cerr << "Error: choose only one of -g, -gp, -lb for coloring\n";
                return 1;
            }
            color_mode = COLOR_LUBY;
        } else if (arg == "-b") {
            if (kernel_mode != KERNEL_NONE) {
                std::cerr << "Error: choose only one of -b, -l for kernel\n";
                return 1;
            }
            kernel_mode = KERNEL_BURKE;
        } else if (arg == "-l") {
            if (kernel_mode != KERNEL_NONE) {
                std::cerr << "Error: choose only one of -b, -l for kernel\n";
                return 1;
            }
            kernel_mode = KERNEL_LAPLACIAN;
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            return 1;
        }
    }

    if (n <= 0 || p < 0.0 || p > 1.0 ||
        color_mode == COLOR_NONE || kernel_mode == KERNEL_NONE) {
        std::cerr << "Usage:\n"
                  << "  ./baseline -n <nodes> -p <prob> -t <threads> -i <iters>\n"
                  << "             (-g | -gp | -lb)  (-b | -l)\n"
                  << "    -g  : greedy (sequential) coloring\n"
                  << "    -gp : speculative parallel greedy coloring\n"
                  << "    -lb  : Luby MIS coloring\n"
                  << "    -b  : Burke refinement kernel\n"
                  << "    -l  : Laplacian smoothing kernel\n";
        return 1;
    }

    omp_set_num_threads(num_threads);

    std::cout << "Generating random graph: n=" << n << " p=" << p
              << " threads=" << num_threads << "\n";
    Graph g = generate_random_graph(n, p);

    // ---- choose coloring algorithm ----
    std::vector<int> color;
    int num_colors = 0;
    double t0 = omp_get_wtime();

    if (color_mode == COLOR_GREEDY) {
        num_colors = greedy_color(g, color, metrics);
    } else if (color_mode == COLOR_SPECULATIVE) {
        num_colors = speculative_parallel_greedy_color(g, color, num_threads, metrics);
    } else if (color_mode == COLOR_LUBY) {
        num_colors = luby_parallel_mis_color(g, color, num_threads);
    }

    double t1 = omp_get_wtime();
    double t_color = t1 - t0;

    if (max_colors > 0 && num_colors > max_colors) {
    for (int &c : color) {
        if (c >= 0) c = c % max_colors;
    }
    num_colors = max_colors;
    }
    long long conf_after = total_conflicts(g, color);

    std::cout << "Coloring: mode="
              << (color_mode == COLOR_GREEDY ? "greedy" :
                  color_mode == COLOR_SPECULATIVE ? "spec_greedy" : "luby")
              << ", colors=" << num_colors
              << ", time=" << t_color
              << " s, conflicts=" << conf_after << "\n";

    double locality_before = compute_color_class_locality(color, num_colors);
    metrics.color_locality_score = locality_before; 
    // ---- run kernel ----
    if (kernel_mode == KERNEL_BURKE) {
        double t2 = omp_get_wtime();
        burke_refine(g, color, num_colors, iters);
        double t3 = omp_get_wtime();
        double t_burke = t3 - t2;
        long long conf_after_burke = total_conflicts(g, color);

        double locality_after = compute_color_class_locality(color, num_colors);

        std::cout << "Burke kernel: time=" << t_burke << " s, conflicts=" << conf_after_burke << "\n";
        std::cout << "Color-class locality (before Burke): " << locality_before << "\n";
        std::cout << "Color-class locality (after Burke):  " << locality_after  << "\n";
        metrics.color_locality_score = locality_after;

    } else if (kernel_mode == KERNEL_LAPLACIAN) {
        auto color_classes = build_color_classes(color, num_colors);

        double total_std = 0.0;
        int nonempty = 0;
        for (const auto &cls : color_classes) {
            if (cls.empty()) continue;
            nonempty++;
            double mean = 0.0;
            for (int v : cls) mean += v;
            mean /= cls.size();
            double var = 0.0;
            for (int v : cls) {
                double d = v - mean;
                var += d * d;
            }
            var /= cls.size();
            total_std += std::sqrt(var);
        }
        if (nonempty > 0) {
            metrics.color_locality_score = total_std / nonempty;
        } else {
            metrics.color_locality_score = 0.0;
        }

        int n_local = g.n;
        std::vector<double> values(n_local);
        std::mt19937 rng(123);
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        for (int i = 0; i < n_local; ++i) values[i] = dist(rng);

        double t4 = omp_get_wtime();
        laplacian_smoothing_color_batches(g, color_classes, values, iters, metrics);
        double t5 = omp_get_wtime();
        double t_lap = t5 - t4;

        std::cout << "Laplacian kernel: time=" << t_lap << " s\n";
    }

    std::cout << "=== Metrics ===\n";
        if (metrics.color_edge_visits > 0) {
            double avg_dist = (double)metrics.color_neigh_dist_sum / (double)metrics.color_edge_visits;
            std::cout << "Coloring edge visits: " << metrics.color_edge_visits  << ", avg |u-v|: " << avg_dist << "\n";
        }
        if (metrics.lap_edge_visits > 0) {
            double avg_dist_lap = (double)metrics.lap_neigh_dist_sum / (double)metrics.lap_edge_visits;
            std::cout << "Laplacian edge visits: " << metrics.lap_edge_visits << ", avg |u-v|: " << avg_dist_lap << "\n";
        }
        std::cout << "Color-class locality score (avg stddev of vertex IDs per color): " << metrics.color_locality_score << "\n\n";

    return 0;
}
