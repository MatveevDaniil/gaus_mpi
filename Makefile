gaus: main.o gaus.o read_print.o
		g++ main.o gaus.o read_print.o -o gaus

main.o: main.cpp
		g++ -c main.cpp

factorial.o: gaus.cpp
		g++ -c gaus.cpp

hello.o: read_print.cpp
		g++ -c read_print.cpp

clean:
		rm -rf *.o hello
