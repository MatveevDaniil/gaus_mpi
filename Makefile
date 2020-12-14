gaus: main.o gaus.o read_print.o
		mpiicc  main.o gaus.o read_print.o -o gaus

main.o: main.cpp
		mpiicc  -c main.cpp

gaus.o: gaus.cpp
		mpiicc  -c gaus.cpp

read_print.o: read_print.cpp
		mpiicc  -c read_print.cpp

clean:
		rm -rf *.o hello
