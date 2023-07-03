/* Copyright 2020 Guanyu Feng, Tsinghua University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cstdio>
#include <cstdint>
#include <vector>
#include <stdio.h>
#include <algorithm>
#include <sstream>
#include <fstream>
#include <iostream>
// int main()
// {
//     std::vector<std::pair<uint64_t, uint64_t>> edges;
//     uint64_t x, y;
//     scanf("%lu%lu", &x, &y) != EOF;
//     printf("%lu, %lu", x, y);    
//     // while(scanf("%lu%lu", &x, &y) != EOF)
//     // {
//     //     edges.emplace_back(x, y);
//     // }
//     // std::random_shuffle(edges.begin(), edges.end());
//     // printf("random done");
//     // for(auto p : edges)
//     // {
//     //     fwrite(&p.first, sizeof(p.first), 1, stdout);
//     //     fwrite(&p.second, sizeof(p.second), 1, stdout);
//     // }
// }
int main(int argc, char *argv[])
{
    // uint64_t x, y;
    // printf("hello");    
    // while(scanf("%lu%lu", &x, &y) != EOF)
    // {
    //     printf("hello");   
    //     fwrite(&x, sizeof(x), 1, stdout);
    //     fwrite(&y, sizeof(x), 1, stdout);
    //     printf("x is %d, y is %d", x, y);
    // }
    if (argc != 3) {
        std::cerr << "Usage: program input_file output_file\n";
        return 1;
    }

    std::ifstream inputFile(argv[1]);
    std::ofstream outputFile(argv[2], std::ios::binary);

    if (!inputFile) {
        std::cerr << "Error opening input file\n";
        return 1;
    }

    if (!outputFile) {
        std::cerr << "Error opening output file\n";
        return 1;
    }

    std::string line;
    std::vector<std::pair<uint64_t, uint64_t>> edges;
    while (getline(inputFile, line)) {
        std::istringstream lineStream(line);
        uint64_t x, y;

        // Try to read two integers
        if (lineStream >> x >> y) {
            // Check if we've reached the end of the line.
            // If not, the line contains more than two numbers and we ignore it.
            std::string remainder;
            if (!(lineStream >> remainder)) {
                // outputFile.write(reinterpret_cast<const char*>(&x), sizeof(x));
                // outputFile.write(reinterpret_cast<const char*>(&y), sizeof(y));
                edges.emplace_back(x,y);
            }
        }
    }
    std::random_shuffle(edges.begin(), edges.end());
    for (auto edge: edges)
    {
        outputFile.write(reinterpret_cast<const char*>(&edge.first), sizeof(edge.first));
        outputFile.write(reinterpret_cast<const char*>(&edge.second), sizeof(edge.second));    
    }
    
    return 0;     
}