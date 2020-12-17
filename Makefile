gaus: main.o gaus.o read_print.o
		mpic++ main.o gaus.o read_print.o -o gaus -O3

main.o: main.cpp
		mpic++ -c main.cpp -O3

gaus.o: gaus.cpp
		mpic++ -c gaus.cpp -O3

read_print.o: read_print.cpp
		mpic++ -c read_print.cpp -O3

clean:
		rm -rf *.o hello
