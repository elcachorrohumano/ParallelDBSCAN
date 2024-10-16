#include <omp.h>
#include <iostream>
#include <string>
#include <fstream>
#include <cmath>
#include <vector>
#include <algorithm>

using namespace std;


// Function to calculate Euclidean distance between two points
float euclidean_distance(float* point1, float* point2) {
    return sqrt(pow(point1[0] - point2[0], 2) + pow(point1[1] - point2[1], 2));
}

//podemos paralelizar aqui en sacar la distancia euclideana de este vector
//podemos usar el sections o usamos el for schedule, chunk, pool_size


// Function to find neighbors within epsilon distance
vector<int> region_query(float** points, int point_idx, float epsilon, long long int size) {
    vector<int> neighbors;
    for (long long int i = 0; i < size; i++) {
        if (euclidean_distance(points[point_idx], points[i]) <= epsilon) {
            neighbors.push_back(i);
        }
    }
    return neighbors;
}


// DBSCAN algorithm
void dbscanParalel(float** points, float epsilon, int min_samples, long long int size, int num_threads, double percentage) {
    vector<int> cluster_labels(size, -1); // -1 for unvisited, 0 for noise, >0 for cluster number
    int cluster_id = 0;

    int chunk_size = ceil(size * percentage);

    omp_set_num_threads(num_threads);
    #pragma omp parallel for schedule(dynamic, chunk_size)
    for (long long int i = 0; i < size; i++) 
    {
        if (cluster_labels[i] != -1) continue;

        vector<int> neighbors = region_query(points, i, epsilon, size);

        if (neighbors.size() < min_samples) {
            cluster_labels[i] = 0; // Mark as noise
            continue;
        }

        cluster_id++;
        cluster_labels[i] = cluster_id;

        vector<int> seed_set(neighbors.begin(), neighbors.end());
        seed_set.erase(remove(seed_set.begin(), seed_set.end(), i), seed_set.end());

        for (size_t j = 0; j < seed_set.size(); j++) {
            int current_point = seed_set[j];

            if (cluster_labels[current_point] == 0) {
                cluster_labels[current_point] = cluster_id;
            }

            if (cluster_labels[current_point] != -1) continue;

            cluster_labels[current_point] = cluster_id;

            vector<int> current_neighbors = region_query(points, current_point, epsilon, size);

            if (current_neighbors.size() >= min_samples) {
                seed_set.insert(seed_set.end(), current_neighbors.begin(), current_neighbors.end());
            }
        }
    }
    // Update points array with cluster information
    for (long long int i = 0; i < size; i++) {
        points[i][2] = cluster_labels[i] > 0 ? 1 : 0; // 1 for core point, 0 for noise
    }

}


// DBSCAN algorithm
void dbscan(float** points, float epsilon, int min_samples, long long int size) {
    vector<int> cluster_labels(size, -1); // -1 for unvisited, 0 for noise, >0 for cluster number
    int cluster_id = 0;

    //for 

    double time1 = omp_get_wtime();

    for (long long int i = 0; i < size; i++) {
        if (cluster_labels[i] != -1) continue;

        vector<int> neighbors = region_query(points, i, epsilon, size);

        if (neighbors.size() < min_samples) {
            cluster_labels[i] = 0; // Mark as noise
            continue;
        }

        cluster_id++;
        cluster_labels[i] = cluster_id;

        vector<int> seed_set(neighbors.begin(), neighbors.end());
        seed_set.erase(remove(seed_set.begin(), seed_set.end(), i), seed_set.end());

        for (size_t j = 0; j < seed_set.size(); j++) {
            int current_point = seed_set[j];

            if (cluster_labels[current_point] == 0) {
                cluster_labels[current_point] = cluster_id;
            }

            if (cluster_labels[current_point] != -1) continue;

            cluster_labels[current_point] = cluster_id;

            vector<int> current_neighbors = region_query(points, current_point, epsilon, size);

            if (current_neighbors.size() >= min_samples) {
                seed_set.insert(seed_set.end(), current_neighbors.begin(), current_neighbors.end());
            }
        }
    }

    // Update points array with cluster information
    for (long long int i = 0; i < size; i++) {
        points[i][2] = cluster_labels[i] > 0 ? 1 : 0; // 1 for core point, 0 for noise
    }

    double time2 = omp_get_wtime();

    double time = time2 - time1;

    cout << " time: " << time << endl;
}

void load_CSV(string file_name, float** points, long long int size) {
    ifstream in(file_name);
    if (!in) {
        cerr << "Couldn't read file: " << file_name << "\n";
    }
    long long int point_number = 0; 
    while (!in.eof() && (point_number < size)) {
        char* line = new char[12];
        streamsize row_size = 12;
        in.read(line, row_size);
        string row = line;
        //cout << stof(row.substr(0, 5)) << " - " << stof(row.substr(6, 5)) << "\n";
        points[point_number][0] = stof(row.substr(0, 5));
        points[point_number][1] = stof(row.substr(6, 5));
        point_number++;
    }
}

void save_to_CSV(string file_name, float** points, long long int size) {
    fstream fout;
    fout.open(file_name, ios::out);
    for (long long int i = 0; i < size; i++) {
        fout << points[i][0] << ","
             << points[i][1] << ","
             << points[i][2] << "\n";
    }
}

int main(int argc, char** argv) {

    float percentage =  0.05;
    long int size[8] = {20000, 40000, 80000, 120000, 140000, 160000, 180000, 200000};
    int threads[4] = {1, 6, 11, 22};

    const float epsilon = 0.03;
    const int min_samples = 10;

    ofstream result_file("results.csv");

    if (!result_file.is_open()) {
        cerr << "Failed to open results file." << endl;
        return 1;
    }

    result_file << "n,n_threads,chunk_size,time\n";

    for (long int s = 0; s < 8; s++) {
        for (int t = 0; t < 4; t++) {
            for (int i = 0; i < 10; i++) {

                double start = omp_get_wtime();
                const string input_file_name = "points/" + to_string(size[s]) + "_data.csv";

                float** points = new float*[size[s]];
                for (long int j = 0; j < size[s]; j++) {
                    points[j] = new float[2];
                }

                load_CSV(input_file_name, points, size[s]);

                dbscanParalel(points, epsilon, min_samples, size[s], threads[t], percentage);

                double end = omp_get_wtime();
                double time = end - start;
                int cz = ceil(size[s] * percentage);

                cout << "n: " << size[s] << ", " << "n_threads: " << threads[t] << ", " << "chunk_size: " << cz << ", " << "time: " << time << "\n" << endl;
                result_file << size[s] << "," << threads[t] << "," << cz << "," << time << "\n";

                // Free allocated memory for points
                for (long int j = 0; j < size[s]; j++) {
                    delete[] points[j];
                }
                
                delete[] points;
            }
            
        }
    }

    result_file.close();

    return 0;
}