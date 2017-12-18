LISTA="72 100 162 202 153 181 158 115 45 35 65 113 133 54 165" 
for var in $LISTA
do 
    ./build/getDescriptor $var 
done
