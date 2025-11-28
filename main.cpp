enum Mode {
    MODE_NONE = 0,
    MODE_GREEDY,
    MODE_BURKE,
    MODE_LAPLACIAN
};

void print_usage(const char* prog) {
    std::cerr <<
        "Usage: " << prog << " -n <nodes> -p <prob> [-t threads] [-i iters] (-g | -b | -l)\n"
        "  -n <nodes>     : number of vertices\n"
        "  -p <prob>      : edge probability (0..1) for Erdos-Renyi graph\n"
        "  -t <threads>   : OpenMP threads (default 4)\n"
        "  -i <iters>     : iterations for Burke (-b) or Laplacian (-l) (default 10)\n"
        "  -g             : run greedy only (report colors + conflicts)\n"
        "  -b             : run Burke refinement (on top of greedy)\n"
        "  -l             : run Laplacian smoothing (on top of greedy coloring)\n";
}

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
                print_usage(argv[0]);
                return 1;
            }
            n = std::atoi(argv[++i]);
        } else if (arg == "-p") {
            if (i + 1 >= argc) {
                std::cerr << "Error: -p requires a value\n";
                print_usage(argv[0]);
                return 1;
            }
            p = std::atof(argv[++i]);
        } else if (arg == "-t") {
            if (i + 1 >= argc) {
                std::cerr << "Error: -t requires a value\n";
                print_usage(argv[0]);
                return 1;
            }
            num_threads = std::atoi(argv[++i]);
        } else if (arg == "-i") {
            if (i + 1 >= argc) {
                std::cerr << "Error: -i requires a value\n";
                print_usage(argv[0]);
                return 1;
            }
            iters = std::atoi(argv[++i]);
        } else if (arg == "-g") {
            if (mode != MODE_NONE) {
                std::cerr << "Error: specify only one of -g, -b, or -l.\n";
                print_usage(argv[0]);
                return 1;
            }
            mode = MODE_GREEDY;
        } else if (arg == "-b") {
            if (mode != MODE_NONE) {
                std::cerr << "Error: specify only one of -g, -b, or -l.\n";
                print_usage(argv[0]);
                return 1;
            }
            mode = MODE_BURKE;
        } else if (arg == "-l") {
            if (mode != MODE_NONE) {
                std::cerr << "Error: specify only one of -g, -b, or -l.\n";
                print_usage(argv[0]);
                return 1;
            }
            mode = MODE_LAPLACIAN;
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    if (n <= 0 || p < 0.0 || p > 1.0 || mode == MODE_NONE) {
        print_usage(argv[0]);
        return 1;
    }

    omp_set_num_threads(num_threads);

    std::cout << "Generating random graph: n=" << n << " p=" << p
              << " threads=" << num_threads << "\n";
    Graph g = generate_random_graph(n, p);

    std::vector<int> color;
    double t0 = omp_get_wtime();
    int num_colors = greedy_color(g, color);
    double t1 = omp_get_wtime();

    std::cout << "Greedy coloring used " << num_colors
              << " colors (time: " << (t1 - t0) << " s)\n";
    long long conf = total_conflicts(g, color);
    std::cout << "Conflicts after greedy: " << conf << "\n";

    if (mode == MODE_GREEDY) {
        return 0;
    }

    if (mode == MODE_BURKE) {
        std::cout << "Running Burke refinement for " << iters << " iterations\n";
        double t2 = omp_get_wtime();
        burke_refine(g, color, num_colors, iters);
        double t3 = omp_get_wtime();

        long long conf_after = total_conflicts(g, color);
        std::cout << "Burke refinement time: " << (t3 - t2) << " s\n";
        std::cout << "Conflicts after Burke: " << conf_after << "\n";
        return 0;
    }

    if (mode == MODE_LAPLACIAN) {
        std::cout << "Running Laplacian smoothing for " << iters << " iterations\n";

        auto color_classes = build_color_classes(color, num_colors);

        int n_local = g.n;
        std::vector<double> values(n_local);
        std::mt19937 rng(123);
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        for (int i = 0; i < n_local; ++i) values[i] = dist(rng);

        double t4 = omp_get_wtime();
        laplacian_smoothing_color_batches(g, color_classes, values, iters);
        double t5 = omp_get_wtime();

        std::cout << "Laplacian smoothing time: " << (t5 - t4) << " s\n";
        std::cout << "vertex values after smoothing (first 10):\n";
        for (int i = 0; i < std::min(n_local, 10); ++i) {
            std::cout << "v " << i << " = " << values[i] << "\n";
        }
        return 0;
    }

    return 0;
}
