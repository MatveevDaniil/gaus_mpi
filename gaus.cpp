#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <mpi.h>
#include "gaus.h"
#include "read_print.h"

int gaus(int n, double* inp, double* rev, double* frev, 
         double* buf, int* index, int rank, int p) {
  int loc_i, rows, max_idx[2];
  double tmp;
  struct rank_idx normInd, globalNormInd;
  MPI_Status st;

  if (rank + 1 > n % p)
    rows = n / p;
  else
    rows = n / p + 1;

  for (int i = 0; i < n; i++) {
    int maxStr = (i % p <= rank) ? i - i % p + rank : i + (p - i % p) + rank;
    int maxCol = i;
    double maxNorm = 0;

    normInd.rank = 0;
    max_idx[0] = -1; 
    max_idx[1] = -1; 

    for (int s = maxStr; s < n; s += p) {
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
    MPI_Allreduce(&num, &globalNum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&err, &globalErr, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    if (globalNum == globalErr)
      return 0;

    MPI_Allreduce(&normInd, &globalNormInd, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);
    MPI_Bcast(&max_idx, 2, MPI_INT, globalNormInd.rank, MPI_COMM_WORLD);
    //if(rank == 0) {
    //  _print_matrix(10, 10, 10, inp);
    //  printf("final %d |%d %d| %lf\n", i, max_idx[0], max_idx[1], globalNormInd.norm);
    //}
    //if (rank == 0) {
    //  printf("%d %d %d\n", max_idx[0], max_idx[1], i);
    //  fflush(stdout);
    //}
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
    if ((max_idx[0] != i) && (max_idx[0] != -1)) { //str
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

    MPI_Bcast(buf, 2 * n, MPI_DOUBLE, i % p, MPI_COMM_WORLD);

    for (int j = 0; j < std::min(rows, n); j++) {
      if (rank == i % p && j == i / p)
        continue;

      tmp = inp[j * n + i];
      for (int k = i; k < n; k++)
        inp[j * n + k] -= tmp * buf[k];
      for (int k = 0; k < n; k++)
        rev[j * n + k] -= tmp * buf[k + n];
    }

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

  for (int i = n - 1; i >= 1; i--) {
    if (rank == i % p)
      for (int k = 0; k < n; k++)
        buf[k] = rev[i / p * n + k];

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

  for (int i = 0; i < n; i++) {
    if (i % p == rank)
      for (int j = 0; j < n; j++)
        frev[i * n + j] = rev[i / p * n + j];
    else
      for (int j = 0; j < n; j++)
        frev[i * n + j] = 0;
  }

  MPI_Allreduce(frev, buf, n * n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      frev[index[i] * n + j] = buf[i * n + j];

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
