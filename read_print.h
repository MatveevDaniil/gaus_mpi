#include <cmath>
#include <iostream>
#include <cstdio>
#include <mpi.h>

void fill_matrix(int k, double* matrix, int n, double* buf,
                 int cur_process, int p);
void _print_matrix(int m, int l, int n, double* matrix);
void print_matrix(int m, int l, int n, double* matrix, 
                  double* buf, int cur_process, int p);
int read_matrix(const char* filename, double* matrix, int n,
                double* buf, int cur_process, int p);