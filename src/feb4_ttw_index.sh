echo "SSWP begin"
./index-sswp ~/graph/ttw/ttw_snap ../../Glign-AE/query_input/TW_queries.txt 16 0.01
echo "WCC begin"
./index-wcc ~/graph/ttw/ttw_snap ../../Glign-AE/query_input/TW_queries.txt 16 0.01

