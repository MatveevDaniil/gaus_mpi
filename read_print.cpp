#include "read_print.h"

inline static double f(int k, int n, int i, int j) {
  switch (k) {
  case 1:
    return (double)n - std::max(i, j) + 1;
  case 2:
    return (double)std::max(i, j);
  case 3:
    return (double)abs(i - j);
  case 4:
    return 1.f / (i + j + 1);
  case 5:
    return (double)i == j;
  default:
    return 0;
  }
}

int read_matrix(const char* filename, double* matrix, int n,
                double* buf, int cur_process, int p) {
  int rows;
  if ((cur_process + 1) > (n % p))
    rows = n / p;
  else
    rows = n / p + 1;
  bool err = false;
  int idx_process;
  FILE* data_file = fopen(filename, "r");
  if (!data_file)
    err = true;

  MPI_Bcast(&err, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (err)
    return -1;

  for (int i = 0; i < n; i++) {
    idx_process = i % p;

    if (cur_process == 0) {
      for (int j = 0; j < n; j++) {
        if (fscanf(data_file, "%lf", buf + j) != 1) {
          err = true;
          break;
        }
        buf[n + j] = (j == i) ? 1 : 0;
      }

      if (idx_process != 0)
        MPI_Send(buf, 2 * n, MPI_DOUBLE, idx_process, 0, MPI_COMM_WORLD);
    }
    else if (cur_process == idx_process) {
      MPI_Status st;
      MPI_Recv(buf, 2 * n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &st);
    }

    if (cur_process == idx_process)
      for (int j = 0; j < n; j++)
        matrix[i / p * n + j] = buf[j];
  }

  if (cur_process == 0)
    if (data_file != 0)
      fclose(data_file);

  MPI_Bcast(&err, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (err)
    return -2;

  return 0;
}

void fill_matrix(int k, double* matrix, int n, double* buf,
                 int cur_process, int p) {
  int rows;
  if ((cur_process + 1) > (n % p))
    rows = n / p;
  else
    rows = n / p + 1;
  for (int i = 0; i < rows; i++)
    for (int j = 0; j < n; j++)
      matrix[i * n + j] = f(k, n, cur_process + p * i, j);
}

void _print_matrix(int m, int l, int n, double* matrix) {
  int i, j;

  for (i = 0; i < std::min(m, l); i++) {
    for (j = 0; j < std::min(m, l); j++) {
      printf(" %lf", matrix[i * n + j]);
      fflush(stdout);
    }
    printf("\n");
    fflush(stdout);
  }
}

void print_matrix(int m, int l, int n, double* matrix,
                  double* buf, int cur_process, int p) {
  int idx_process;
  int rng = std::min(std::min(m, l), n / p);
  for (int i = 0; i < rng; i++) {
    idx_process = i % p;

    if (cur_process == idx_process)
      memcpy(buf, matrix + i / p * n, n * sizeof(double));

    if (cur_process == idx_process && cur_process != 0)
      MPI_Send(buf, n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    else if (cur_process == 0) {
      MPI_Status st;
      if (idx_process != 0)
        MPI_Recv(buf, n, MPI_DOUBLE, idx_process, 0, MPI_COMM_WORLD, &st);
      for (int j = 0; j < rng; j++) {
        printf("%f ", buf[j]);
        fflush(stdout);
      }
      printf("\n");
      fflush(stdout);
    }
  }
}
