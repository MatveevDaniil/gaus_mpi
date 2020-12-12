struct rank_idx { double norm; int rank; };

double count_residual(double* inp, double* rev, int n,
                      double* buf, int rank, int p);
int gaus(int n, double* a, double* b, double* x,
         double* buf, int* index, int rank, int p);