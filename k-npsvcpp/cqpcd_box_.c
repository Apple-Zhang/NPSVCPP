#include <mex.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

/*
    Coordinate desent method for Quadratic programming under box constraints.
    Assumption: Q is semi positive definite.

    Written by Apple Zhang, 2023.
*/

inline int max(const int a, const int b) {
    return a > b ? a : b;
}

inline double dot(const double* a, const double* b, int n) {
    // early return, avoid loops
    if (n <= 0) return 0;
    switch (n) {
        case 1: return a[0] * b[0];
        case 2: return a[0] * b[0] + a[1] * b[1];
        case 3: return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
        case 4: return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
    }

    register double sum = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
    for (int i = 4; i < n; i++) {
        sum += a[i] * b[i];
    }

    return sum;
}

inline double clamp(const double a, const double lb, const double ub) {
    if      (a > ub) return ub;
    else if (a < lb) return lb;
    else             return a;
}

inline void shuffle(int *array, int n)
{
    // actually, it is a ''determined'' random seed. The reason is two-sided.
    // 1. Our aim is to shuffle, but not generate completely random result.
    // 2. For problem in the same size, the random sequence will be the same, convenient for reproducable.
    srand(array[rand() % n]); 
    int i;
    for (i = 0; i < n-1; i++) 
    {
        int j = i + rand() / (RAND_MAX / (n - i) + 1);
        int t = array[j];
        array[j] = array[i];
        array[i] = t;
    }
}

inline int checkKKTOptimal(const double *Q, const double *h, const double lb, const double ub, const int n, 
                           const int tol,   const double *alpha) 
{
    const double* qptr = Q;
    double        grad;
    int           isKKToptimal = 1;
    for (int i = 0; i < n; i++) {
        grad = dot(alpha, qptr, n) + h[i];
        if      (alpha[i] == ub) isKKToptimal &= (grad < 0);
        else if (alpha[i] == lb) isKKToptimal &= (grad > 0);
        else                     isKKToptimal &= (fabs(grad) <= tol);
        if      (!isKKToptimal) return 0;
        qptr += n;
    }
    return 1;
}

void qpsor(const double *Q, const double *h, const double lb, const double ub, const int n,
           const double tol, double *alpha, int isWarm)
{
    const int max_iter = 300;
    double*   weight;
    int*      index;
    
    register double grad;
    const double*   qptr;
    int             isKKToptimal;
    int             checkKKTFreq;

    // check KKT optimality of the warm-start
    // early stop if the KKT optimal condition is satisfied.
    if (isWarm && checkKKTOptimal(Q, h, lb, ub, n, tol, alpha)) {
        return;
    }

    weight = (double*)malloc(sizeof(double)*n);
    index  = (int*)malloc(sizeof(int)*n);
    
    qptr = Q;
    for (int i = 0; i < n; i++) {
        // cache omega/Q[i,i] in weight
        weight[i] = 1 / *qptr;
        qptr += (n + 1);
        
        // prepare for randperm (shuffle)
        index[i] = i;
    }

    // main loop
    checkKKTFreq = 64; // denotes the frequency that check KKT optimality
    for (int lp = 0; lp < max_iter; lp++) {
        isKKToptimal = 1;

        // shuffle the iteration order for faster convergence
        shuffle(index, n);

        // update alpha
        register double temp;
        register int    idx;
        for (int i = 0; i < n; i++) {
            // get shuffled index
            idx = index[i];

            // projected coordinate desdent
            grad = dot(alpha, Q + idx*n, n) + h[idx];
            temp = alpha[idx] - weight[idx] * grad;
            temp = clamp(temp, lb, ub);

            // update
            alpha[idx] = temp;
        }

        /* check KKT optimality */
        if ((lp+1) % checkKKTFreq == 0) {
            if (checkKKTOptimal(Q, h, lb, ub, n, tol, alpha)) {
                break;
            }
            checkKKTFreq = max(checkKKTFreq >> 1, 1); // half the frequency
            break;
        }
    }

    free(weight);
    free(index);
}

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    double* q;
    double* h;
    double lb;
    double ub;
    double omg;
    double eps;

    double* alpha;
    double* warm;
    int isWarm;

    // check arguments;
    if (nrhs < 5 || nrhs > 6) {
        mexErrMsgIdAndTxt("MLAI:cqpsor:InvalidInputNumber", "Usage: alpha = cqpsor_box(Q, h, lb, ub, eps[,warm_start]).");
    }
    if (nlhs != 1) {
        mexErrMsgIdAndTxt("MLAI:cqpsor:InvalidOutputNumber", "Need ONE output arguments");
    }
    isWarm = (nrhs == 6);
    
    /* check validation of Q is type double */
    if (!mxIsDouble(prhs[0]) || mxIsComplex(prhs[0]) || mxIsSparse(prhs[0])) {
        mexErrMsgIdAndTxt("MLAI:cqpsor:notDoubleQ", "Input matrix Q must be full double (sparse is not supported yet).");
    }

    size_t m = mxGetM(prhs[0]);
    size_t n = mxGetN(prhs[0]);

    // check square Q
    if (m != n) { 
        mexErrMsgIdAndTxt("MLAI:cqpsor:QNotSquare", "Q must be a square matrix");
    }

    /* check validation of Q is type double */
    if (!mxIsDouble(prhs[1]) || mxIsComplex(prhs[1]) || mxIsSparse(prhs[1])) {
        mexErrMsgIdAndTxt("MLAI:cqpsor:notDoubleH", "Input vector h must be full double (sparse is not supported yet).");
    }
    if (mxGetM(prhs[1]) > 1 && mxGetN(prhs[1]) > 1) {
        mexErrMsgIdAndTxt("MLAI:cqpsor:notVectorH", "Input vector h should be a vector but not matrix.");
    }

    /* make sure the LB input argument is type double */
    if (!mxIsScalar(prhs[2]) || !mxIsDouble(prhs[2]) || mxIsComplex(prhs[2])) {
        mexErrMsgIdAndTxt("MLAI:cqpsor:notDoubleLB", "lower bound must be a double sclar.");
    }

    /* make sure the UB input argument is type double */
    if (!mxIsScalar(prhs[3]) || !mxIsDouble(prhs[3]) || mxIsComplex(prhs[3])) {
        mexErrMsgIdAndTxt("MLAI:cqpsor:notDoubleUB", "upper bound must be a double sclar.");
    }

    /* make sure the omega input argument is type double */
    if (!mxIsScalar(prhs[4]) || !mxIsDouble(prhs[4]) || mxGetScalar(prhs[4]) < 0) {
        mexErrMsgIdAndTxt("MLAI:cqpsor:notDoubleTOL", "Input tolerance must be a positive double sclar.");
    }
    if (isWarm) {
        if (mxGetM(prhs[5]) > 1 && mxGetN(prhs[5]) > 1 || mxGetM(prhs[5]) != m && mxGetN(prhs[5]) != m) {
            mexErrMsgIdAndTxt("MLAI:cqpsor:notValidWarmStart", "The warm-start should be a vector in the same size with h.");
        }
    }

    // output vector
    plhs[0] = mxCreateDoubleMatrix(n, 1, mxREAL);

#if MX_HAS_INTERLEAVED_COMPLEX
    q = mxGetDoubles(prhs[0]);
    h = mxGetDoubles(prhs[1]);
    lb = mxGetScalar(prhs[2]);
    ub = mxGetScalar(prhs[3]);
    eps = mxGetScalar(prhs[4]);
    if (isWarm) {
        warm = mxGetDoubles(prhs[5]);
    }

    // results
    alpha = mxGetDoubles(plhs[0]);
#else
    q = mxGetPr(prhs[0]);
    h = mxGetPr(prhs[1]);
    lb = mxGetScalar(prhs[2]);
    ub = mxGetScalar(prhs[3]);
    eps = mxGetScalar(prhs[4]);
    if (isWarm) {
        warm = mxGetPr(prhs[5]);
    }
    
    // results
    alpha = mxGetPr(plhs[0]);
#endif

    if (isWarm) {
        memcpy(alpha, warm, sizeof(double)*n);
    }
    else {
        memset(alpha, 0, sizeof(double)*n);
    }

    qpsor(q, h, lb, ub, (int)n, eps, alpha, isWarm);
}