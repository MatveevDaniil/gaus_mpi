#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include "read_print.h"
#include "gaus.h"

int main(int argc, char** argv) {
  double* inp, * rev, * full_reverse, * buf; // исходная и обратные матрицы inp и rev, full_reverse - чтобы удобно было ошибку считать
  // buf - буфер для отправки сообщений
  int n, m, k; // параметры запуска
  int p, rows, cur_process; // параметры распаралеливания общее количество процессов p и номер текущего процесса cur_process
  int err1 = 0, err2 = 0; // переменные для ошибок проги
  int* index; // нужно было т.к. главный по все матрице, тебе не нужно
  std::string file_name = "";

  // инициализируем mpi - получаем 
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &p);
  MPI_Comm_rank(MPI_COMM_WORLD, &cur_process);
  
  // собираем аргументы запуска
  if (argc < 4 || argc > 5) {
    std::cout << "wrong argument's number\n" << "usage: 'prog n m 0 filename' or 'prog n m k'" << std::endl;
    MPI_Finalize();
    return -1;
  }
  else {
    n = atoi(argv[1]);
    m = atoi(argv[2]);
    k = atoi(argv[3]);
    if (k == 0) {
      if (argc == 5) {
        file_name = argv[4];
      }
      else {
        std::cout << "if k == 0, u need to give file with matrix: 'prog n m 0 filename'" << std::endl;
        MPI_Finalize();
        return -2;
      }
    }
    if (n <= 0 || m <= 0 || k < 0 || k > 6) {
      std::cout << "usage: 'prog n m 0 filename' or 'prog n m k', where n > 0, m > 0, 4 >= k > 0" << std::endl;
      MPI_Finalize();
      return -2;
    }
  }

  // уменьшаем количество процессов если требуется
  if (p > n) {
    printf("Too many threads\n");
    p = n;
    //MPI_Finalize();
    //return -3;
  }
  // количество строк во входной и выходной матрицы, 
  // в зависимости от номера процесса может увеличиться на 1, т.к. n может n не делиться нацело на p
  // входная матрица данного процесса содержит все строки реальной матрицы начиная с cur_process и дальше через p
  if ((cur_process + 1) > (n % p))
    rows = n / p;
  else
    rows = n / p + 1;
  // алоцируем память
  inp = new double[n * rows];
  rev = new double[n * rows];
  full_reverse = new double[n * n];
  buf = new double[n * n];
  index = new int[n];

  if (!(inp && rev && buf && full_reverse && index))
    err1 = 1;
  // собираем ошибки аллокации из всех процессов
  MPI_Allreduce(&err1, &err2, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  // если какой-то процесс не смог выделить память завершаем программу - MPI_Finalize
  if (err2) {
    if (inp) delete[] inp;
    if (rev) delete[] rev;
    if (buf) delete[] buf;
    if (full_reverse) delete[] full_reverse;
    if (index) delete[] index;
    MPI_Finalize();
    return -4;
  }
  // заполняем входную матрицу - либо из файла, либо формулой
  err1 = err2 = 0;
  if (file_name == "")
    fill_matrix(k, inp, n, buf, cur_process, p);
  else
    err1 = read_matrix(file_name.c_str(), inp, n, buf, cur_process, p);
  fill_matrix(5, rev, n, buf, cur_process, p);
  // собираем ошибки
  MPI_Allreduce(&err1, &err2, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  // тебе не нужно
  for (int i = 0; i < n; i++)
    index[i] = i;
  // если какой-то процесс не смог заполнить возвращаем ошибку
  if (err2) {
    delete[] inp, rev, buf, full_reverse, index;
    MPI_Finalize();
    return -5;
  }
  // печатаем входную матрицу
  // чтобы не печаталось много раз текст "input matrix" печатаем только в 0-ом процессе
  if (cur_process == 0) 
    std::cout << "input matrix" << std::endl;
  print_matrix(m, m, n, inp, buf, cur_process, p);
  // основной алгоритм
  // MPI_Barrier - функция для синхронизации процессов
  MPI_Barrier(MPI_COMM_WORLD);
  double time = MPI_Wtime();

  int gaus_algo_error = gaus(n, inp, rev, full_reverse, buf, index, cur_process, p);

  MPI_Barrier(MPI_COMM_WORLD);
  // считаем время, когда все процессы завершились
  time = MPI_Wtime() - time;
  // отправляем полную обратную матрицу из 0-ого во все остальные процессы
  MPI_Bcast(full_reverse, n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  
  // если какой-то процесс словил ошибку алгоритма завершаем работу
  if (!gaus_algo_error) {
    if(cur_process == 0)
      std::cout << "det(matrix) = 0" << std::endl;
    delete[] inp, rev, buf, full_reverse, index;
    MPI_Finalize();
    return -6;
  }
  // печатаем время и обратную матрицу
  if (cur_process == 0) {
    std::cout << "reverse matrix" << std::endl;
    _print_matrix(m, m, n, full_reverse);
  }
  if (cur_process == 0)
    std::cout << "time: " << time / (double)CLOCKS_PER_SEC << " s" << std::endl;
  // заполним заново входную матрицу (я её менял в течении алгоритма)
  if (file_name == "")
    fill_matrix(k, inp, n, buf, cur_process, p);
  else
    err1 = read_matrix(file_name.c_str(), inp, n, buf, cur_process, p);
  // считаем и выводим невязку
  double gaus_id_error = count_residual(inp, full_reverse, n, buf, cur_process, p);
  if (cur_process == 0)
    std::cout << "nerror: " << gaus_id_error << std::endl;
  // всё
  delete[] inp, rev, buf, full_reverse, index;
  MPI_Finalize();
  return 0;
}