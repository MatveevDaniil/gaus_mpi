#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <mpi.h>
#include "gaus.h"
#include "read_print.h"

int gaus(int n, double* inp, double* rev, double* frev, 
         double* buf, int* index, int rank, int p) {
  // rank - номер текущего процесса, p - общее кол-во процессов
  int loc_i, rows, max_idx[2];
  double tmp;
  struct rank_idx normInd, globalNormInd;
  MPI_Status st;
  // определяем количество строк входной матрицы по тому же принципу, что в main
  if (rank + 1 > n % p)
    rows = n / p;
  else
    rows = n / p + 1;
// прямой ход гауса
  for (int i = 0; i < n; i++) {
    // инициализируем переменные для индексов максимального элемента
    // i - i % p - начальный индекс 'блока' по p строк
    // rank или p + rank в зависимости от номера текущего ранга
    int maxStr = (i % p <= rank) ? i - i % p + rank : i + (p - i % p) + rank;
    int maxCol = i;
    double maxNorm = 0;
    // normInd - структурка которая содержит индексы максимального элемента и его модуль
    normInd.rank = 0;
    max_idx[0] = -1; 
    max_idx[1] = -1; 

    for (int s = maxStr; s < n; s += p) {
      // здесь и далее чтобы получить индекс в матрице которая доступна текущему процессу
      // надо поделить номер текущей итерации на количество процессов
      loc_i = s / p;
      for (int q = i; q < n; q++) {
        tmp = fabs(inp[loc_i * n + q]);
        if (tmp > maxNorm) {
          maxNorm = tmp;
          maxStr = s;
          maxCol = q;
        }
      }
    }
    //printf("%d %d, %d %d %lf\n", i, rank, maxStr, maxCol, maxNorm);
    int num = 0, err = 0, globalNum = 0, globalErr = 0;

    int s = (i % p <= rank) ? i - i % p + rank : i + (p - i % p) + rank;
    // делаем проверку, что максимальный элемент не 0
    if (s < n) {
      num = 1;
      max_idx[0] = maxStr;
      max_idx[1] = maxCol;
      normInd.rank = rank;
      normInd.norm = maxNorm;
      if (fabs(maxNorm) < 1e-16)
        err = 1;
    }
    fflush(stdout);
    // собираем ошибки из всех процессов
    MPI_Allreduce(&num, &globalNum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&err, &globalErr, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    if (globalNum == globalErr)
      return 0;
    // собираем максимальные элементы из всех процессов
    MPI_Allreduce(&normInd, &globalNormInd, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);
    MPI_Bcast(&max_idx, 2, MPI_INT, globalNormInd.rank, MPI_COMM_WORLD);
  // переставляем столбцы, тут не нужно обмениваться сообщениями, т.к. у все элементы которые надо переставлять доступны процессу
    if ((max_idx[1] != i) && (max_idx[0] != -1)) { //col
      for (int j = 0; j < std::min(rows, n); j++) {
        tmp = inp[j * n + i];
        inp[j * n + i] = inp[j * n + max_idx[1]];
        inp[j * n + max_idx[1]] = tmp;
      }

      tmp = index[i];
      index[i] = index[max_idx[1]];
      index[max_idx[1]] = tmp;
    }
  // переставляем строки, тут уже придётся обмениваться сообщениями, т.к. максимальная строка может быть у другого процесса
    if ((max_idx[0] != i) && (max_idx[0] != -1)) { //str
      // если максимальная строка у этого же процесса, то просто переставляем строки
      if ((globalNormInd.rank == (i % p)) && (globalNormInd.rank == rank)) {
        for (int j = 0; j < n; j++) {
          tmp = inp[i / p * n + j];
          inp[i / p * n + j] = inp[max_idx[0] / p * n + j];
          inp[max_idx[0] / p * n + j] = tmp;

          tmp = rev[i / p * n + j];
          rev[i / p * n + j] = rev[max_idx[0] / p * n + j];
          rev[max_idx[0] / p * n + j] = tmp;
        }
      }
      // 2 следующий else if разбираются со случаем, когда максимальные строки у разных процессов
      // MPI_Sendrecv_replace работает сразу на получение и отправку, 
      // сообщение (строку в данном случае) из первого else if он отправит во второй и от него же получит сообщение
      else if (globalNormInd.rank == rank) {
        loc_i = max_idx[0] / p;
        memcpy(buf, inp + loc_i * n, n * sizeof(double));
        memcpy(buf + n, rev + loc_i * n, n * sizeof(double));

        MPI_Sendrecv_replace(buf, 2 * n, MPI_DOUBLE, i % p, 0, i % p, 0, MPI_COMM_WORLD, &st);

        memcpy(inp + loc_i * n, buf, n * sizeof(double));
        memcpy(rev + loc_i * n, buf + n, n * sizeof(double));
      }
      else if ((i % p) == rank) {
        loc_i = i / p;
        memcpy(buf, inp + loc_i * n, n * sizeof(double));
        memcpy(buf + n, rev + loc_i * n, n * sizeof(double));

        MPI_Sendrecv_replace(buf, 2 * n, MPI_DOUBLE, globalNormInd.rank, 0, globalNormInd.rank, 0, MPI_COMM_WORLD, &st);

        memcpy(inp + loc_i * n, buf, n * sizeof(double));
        memcpy(rev + loc_i * n, buf + n, n * sizeof(double));
      }
    }
    // записываем в буфер строку которую потом будем вычитать из остальных
    // здесь и далее проверка rank == i % p проверяет принадлежит ли i-ая строка текущему процессу
    // если не принадлежит, то процесс просто пройдёт мимо
    if (rank == i % p) {
      tmp = 1. / inp[i / p * n + i];
      for (int j = i; j < n; j++) {
        inp[i / p * n + j] *= tmp;
        buf[j] = inp[i / p * n + j];
      }

      for (int j = 0; j < n; j++) {
        rev[i / p * n + j] *= tmp;
        buf[j + n] = rev[i / p * n + j];
      }
    }

    if (i == n - 1)
      continue;
    // отправляем эту строку в буфер всем остальным процессам
    MPI_Bcast(buf, 2 * n, MPI_DOUBLE, i % p, MPI_COMM_WORLD);
    // вычитаем строку из остальных
    for (int j = 0; j < std::min(rows, n); j++) {
      if (rank == i % p && j == i / p)
        continue;

      tmp = inp[j * n + i];
      for (int k = i; k < n; k++)
        inp[j * n + k] -= tmp * buf[k];
      for (int k = 0; k < n; k++)
        rev[j * n + k] -= tmp * buf[k + n];
    }
    // отдельный разбор последней итерации гауса
    if (i == n - 2) {
      err = 0;
      if (rank == (n - 1) % p)
        if (fabs(inp[(n - 1) / p * n + n - 1]) < 1e-16)
          err = 1;

      MPI_Bcast(&err, 1, MPI_INT, (n - 1) % p, MPI_COMM_WORLD);

      if (err)
        return 0;
    }
  }
// обратный ход гауса
  for (int i = n - 1; i >= 1; i--) {
    if (rank == i % p)
      for (int k = 0; k < n; k++)
        buf[k] = rev[i / p * n + k];
    // отправляем вычитаемую строку всем остальным процессам.
    MPI_Bcast(buf, n, MPI_DOUBLE, i % p, MPI_COMM_WORLD);

    int start;
    if (rank < i % p)
      start = i / p;
    else if (i / p - 1 >= 0)
      start = i / p - 1;
    else
      continue;

    for (int j = start; j >= 0; j--)
      for (int k = 0; k < n; k++)
        rev[j * n + k] -= buf[k] * inp[j * n + i];
  }
// хитрожопая схема для обратной перестановки строк матрицы
  for (int i = 0; i < n; i++) 
    if (i % p == rank)
      for (int j = 0; j < n; j++)
        inp[i / p * n + j] = rev[i / p * n + j];
  for (int i = 0; i < n; i++) { // collect full matrix
    if (0 == rank) {
      loc_i = i / p;
      MPI_Sendrecv_replace(inp + loc_i * n, n, MPI_DOUBLE, i % p, 0, i % p, 0, MPI_COMM_WORLD, &st);//MPI_Recv(inp + loc_i * n, n, MPI_DOUBLE, i % p, 0, MPI_COMM_WORLD, &st);
      memcpy(buf + i * n, inp + loc_i * n, n * sizeof(double));
    }
    else if ((i % p) == rank) {
      loc_i = i / p; //MPI_Send(inp + loc_i * n, n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
      MPI_Sendrecv_replace(inp + loc_i * n, n, MPI_DOUBLE, 0, 0, 0, 0, MPI_COMM_WORLD, &st);
    }
  }
  if (rank == 0) {
    for (int i = 0; i < n; i++)
      for (int j = 0; j < n; j++)
        frev[index[i] * n + j] = buf[i * n + j];
  }
  for (int i = 0; i < n; i++) { // collect full matrix
    if (0 == rank) {
      loc_i = i / p;
      memcpy(inp + loc_i * n, frev + i * n, n);
      MPI_Sendrecv_replace(inp + loc_i * n, n, MPI_DOUBLE, i % p, 0, i % p, 0, MPI_COMM_WORLD, &st);
    }
    else if ((i % p) == rank) {
      loc_i = i / p;
      MPI_Sendrecv_replace(inp + loc_i * n, n, MPI_DOUBLE, 0, 0, 0, 0, MPI_COMM_WORLD, &st);
      memcpy(rev + loc_i * n, inp + loc_i * n, n * sizeof(double));
    }
  }

  return 1;
}


double count_residual(double* inp, double* rev, int n, 
                      double* buf, int rank, int p) {
  int i, j, k;
  int rows;
  double residual = 0, max_residual = 0;

  for (i = 0; i < 2 * n; i++)
    buf[i] = 0;

  if (rank + 1 > n % p)
    rows = n / p;
  else
    rows = n / p + 1;

  for (i = 0; i < std::min(rows, n); i++) {
    for (k = 0; k < n; k++)
      for (j = 0; j < n; j++)
        buf[k] += inp[i * n + j] * rev[j * n + k];

    for (j = 0; j < n; j++) {
      buf[i + n] += fabs(buf[j]);
      buf[j] = 0;
    }

    buf[i + n]--;
  }

  for (i = 0; i < std::min(rows, n); i++)
    residual = fmax(residual, buf[i + n]);

  MPI_Allreduce(&residual, &max_residual, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

  return max_residual;
}
