gaus: main.o gaus.o read_print.o
		g++ main.o gaus.o read_print.o -o gaus

main.o: main.cpp
		g++ -c main.cpp

gaus.o: gaus.cpp
		g++ -c gaus.cpp

read_print.o: read_print.cpp
		g++ -c read_print.cpp

clean:
		rm -rf *.o hello
