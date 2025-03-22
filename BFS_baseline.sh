apps=(
    "build/BFSbaseline"
)

graph_sources=(
    "$HOME/graph/sx/stack_snap source/sx.txt"
    "$HOME/graph/wiki/snap_wiki source/wiki_src.txt"
    "$HOME/graph/wen/snap_wen source/wen.txt"
    "$HOME/graph/dl/snap_dl source/dl_src.txt"
    "$HOME/graph/or/snap_or source/or.txt"
    "$HOME/graph/ttw/ttw_snap source/TW_queries.txt"
)

batch_params=(
    "8 0.02"  "16 0.01"  "32 0.005"  "64 0.0025"
)

for app in "${apps[@]}"; do
    for gs in "${graph_sources[@]}"; do
        IFS=" " read -r graphfile root_file <<< "$gs"
        for params in "${batch_params[@]}"; do
            read -r batch_num batch_size <<< "$params"
            for extra_param in {0..15}; do
                echo "Running: $app $graphfile $root_file $batch_num $batch_size $extra_param"
                $app "$graphfile" "$root_file" "$batch_num" "$batch_size" "$extra_param"
            done
        done
    done
done
