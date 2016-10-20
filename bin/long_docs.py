ROUTINE_DOCS = {}

ROUTINE_DOCS["zdscal"] = """
ZDSCAL scales a vector by a constant.
"""

ROUTINE_DOCS["snrm2"] = """
SNRM2 returns the euclidean norm of a vector via the function
name, so that

```text
   SNRM2 := sqrt( x'*x ).
```
"""

ROUTINE_DOCS["cgemv"] = """
CGEMV performs one of the matrix-vector operations

```text
   y := alpha*A*x + beta*y,   or   y := alpha*A**T*x + beta*y,   or
```

```text
   y := alpha*A**H*x + beta*y,
```

where alpha and beta are scalars, x and y are vectors and A is an
m by n matrix.
"""

ROUTINE_DOCS["srot"] = """
applies a plane rotation.
"""

ROUTINE_DOCS["drotmg"] = """
CONSTRUCT THE MODIFIED GIVENS TRANSFORMATION MATRIX H WHICH ZEROS
THE SECOND COMPONENT OF THE 2-VECTOR  (DSQRT(DD1)*DX1,DSQRT(DD2)*>    DY2)**T.
WITH DPARAM(1)=DFLAG, H HAS ONE OF THE FOLLOWING FORMS..

DFLAG=-1.D0     DFLAG=0.D0        DFLAG=1.D0     DFLAG=-2.D0

  (DH11  DH12)    (1.D0  DH12)    (DH11  1.D0)    (1.D0  0.D0)
H=(          )    (          )    (          )    (          )
  (DH21  DH22),   (DH21  1.D0),   (-1.D0 DH22),   (0.D0  1.D0).
LOCATIONS 2-4 OF DPARAM CONTAIN DH11, DH21, DH12, AND DH22
RESPECTIVELY. (VALUES OF 1.D0, -1.D0, OR 0.D0 IMPLIED BY THE
VALUE OF DPARAM(1) ARE NOT STORED IN DPARAM.)

THE VALUES OF GAMSQ AND RGAMSQ SET IN THE DATA STATEMENT MAY BE
INEXACT.  THIS IS OK AS THEY ARE ONLY USED FOR TESTING THE SIZE
OF DD1 AND DD2.  ALL ACTUAL SCALING OF DATA IS DONE USING GAM.
"""

ROUTINE_DOCS["zhpmv"] = """
ZHPMV  performs the matrix-vector operation

```text
   y := alpha*A*x + beta*y,
```

where alpha and beta are scalars, x and y are n element vectors and
A is an n by n hermitian matrix, supplied in packed form.
"""

ROUTINE_DOCS["zcopy"] = """
ZCOPY copies a vector, x, to a vector, y.
"""

ROUTINE_DOCS["dsdot"] = """
Compute the inner product of two vectors with extended
precision accumulation and result.

Returns D.P. dot product accumulated in D.P., for S.P. SX and SY
DSDOT = sum for I = 0 to N-1 of  SX(LX+I*INCX) * SY(LY+I*INCY),
where LX = 1 if INCX .GE. 0, else LX = 1+(1-N)*INCX, and LY is
defined in a similar way using INCY.
"""

ROUTINE_DOCS["zgemm"] = """
ZGEMM  performs one of the matrix-matrix operations

```text
   C := alpha*op( A )*op( B ) + beta*C,
```

where  op( X ) is one of

```text
   op( X ) = X   or   op( X ) = X**T   or   op( X ) = X**H,
```

alpha and beta are scalars, and A, B and C are matrices, with op( A )
an m by k matrix,  op( B )  a  k by n matrix and  C an m by n matrix.
"""

ROUTINE_DOCS["sdot"] = """
SDOT forms the dot product of two vectors.
uses unrolled loops for increments equal to one.
"""

ROUTINE_DOCS["zgerc"] = """
ZGERC  performs the rank 1 operation

```text
   A := alpha*x*y**H + A,
```

where alpha is a scalar, x is an m element vector, y is an n element
vector and A is an m by n matrix.
"""

ROUTINE_DOCS["ssyr2k"] = """
SSYR2K  performs one of the symmetric rank 2k operations

```text
   C := alpha*A*B**T + alpha*B*A**T + beta*C,
```

or

```text
   C := alpha*A**T*B + alpha*B**T*A + beta*C,
```

where  alpha and beta  are scalars, C is an  n by n  symmetric matrix
and  A and B  are  n by k  matrices  in the  first  case  and  k by n
matrices in the second case.
"""

ROUTINE_DOCS["csrot"] = """
CSROT applies a plane rotation, where the cos and sin (c and s) are real
and the vectors cx and cy are complex.
jack dongarra, linpack, 3/11/78.
"""

ROUTINE_DOCS["cgemm"] = """
CGEMM  performs one of the matrix-matrix operations

```text
   C := alpha*op( A )*op( B ) + beta*C,
```

where  op( X ) is one of

```text
   op( X ) = X   or   op( X ) = X**T   or   op( X ) = X**H,
```

alpha and beta are scalars, and A, B and C are matrices, with op( A )
an m by k matrix,  op( B )  a  k by n matrix and  C an m by n matrix.
"""

ROUTINE_DOCS["cscal"] = """
CSCAL scales a vector by a constant.
"""

ROUTINE_DOCS["ctrsv"] = """
CTRSV  solves one of the systems of equations

```text
   A*x = b,   or   A**T*x = b,   or   A**H*x = b,
```

where b and x are n element vectors and A is an n by n unit, or
non-unit, upper or lower triangular matrix.

No test for singularity or near-singularity is included in this
routine. Such tests must be performed before calling this routine.
"""

ROUTINE_DOCS["zhemm"] = """
ZHEMM  performs one of the matrix-matrix operations

```text
   C := alpha*A*B + beta*C,
```

or

```text
   C := alpha*B*A + beta*C,
```

where alpha and beta are scalars, A is an hermitian matrix and  B and
C are m by n matrices.
"""

ROUTINE_DOCS["sscal"] = """
scales a vector by a constant.
uses unrolled loops for increment equal to 1.
"""

ROUTINE_DOCS["dtrsm"] = """
DTRSM  solves one of the matrix equations

```text
   op( A )*X = alpha*B,   or   X*op( A ) = alpha*B,
```

where alpha is a scalar, X and B are m by n matrices, A is a unit, or
non-unit,  upper or lower triangular matrix  and  op( A )  is one  of

```text
   op( A ) = A   or   op( A ) = A**T.
```

The matrix X is overwritten on B.
"""

ROUTINE_DOCS["ssyr"] = """
SSYR   performs the symmetric rank 1 operation

```text
   A := alpha*x*x**T + A,
```

where alpha is a real scalar, x is an n element vector and A is an
n by n symmetric matrix.
"""

ROUTINE_DOCS["ztpmv"] = """
ZTPMV  performs one of the matrix-vector operations

```text
   x := A*x,   or   x := A**T*x,   or   x := A**H*x,
```

where x is an n element vector and  A is an n by n unit, or non-unit,
upper or lower triangular matrix, supplied in packed form.
"""

ROUTINE_DOCS["srotg"] = """
SROTG construct givens plane rotation.
"""

ROUTINE_DOCS["dzasum"] = """
DZASUM takes the sum of the (|Re(.)| + |Im(.)|)'s of a complex vector and
returns a single precision result.
"""

ROUTINE_DOCS["dgemv"] = """
DGEMV  performs one of the matrix-vector operations

```text
   y := alpha*A*x + beta*y,   or   y := alpha*A**T*x + beta*y,
```

where alpha and beta are scalars, x and y are vectors and A is an
m by n matrix.
"""

ROUTINE_DOCS["sswap"] = """
interchanges two vectors.
uses unrolled loops for increments equal to 1.
"""

ROUTINE_DOCS["sgemv"] = """
SGEMV  performs one of the matrix-vector operations

```text
   y := alpha*A*x + beta*y,   or   y := alpha*A**T*x + beta*y,
```

where alpha and beta are scalars, x and y are vectors and A is an
m by n matrix.
"""

ROUTINE_DOCS["stbmv"] = """
STBMV  performs one of the matrix-vector operations

```text
   x := A*x,   or   x := A**T*x,
```

where x is an n element vector and  A is an n by n unit, or non-unit,
upper or lower triangular band matrix, with ( k + 1 ) diagonals.
"""

ROUTINE_DOCS["csscal"] = """
CSSCAL scales a complex vector by a real constant.
"""

ROUTINE_DOCS["zhpr2"] = """
ZHPR2  performs the hermitian rank 2 operation

```text
   A := alpha*x*y**H + conjg( alpha )*y*x**H + A,
```

where alpha is a scalar, x and y are n element vectors and A is an
n by n hermitian matrix, supplied in packed form.
"""

ROUTINE_DOCS["sgemm"] = """
SGEMM  performs one of the matrix-matrix operations

```text
   C := alpha*op( A )*op( B ) + beta*C,
```

where  op( X ) is one of

```text
   op( X ) = X   or   op( X ) = X**T,
```

alpha and beta are scalars, and A, B and C are matrices, with op( A )
an m by k matrix,  op( B )  a  k by n matrix and  C an m by n matrix.
"""

ROUTINE_DOCS["dsbmv"] = """
DSBMV  performs the matrix-vector  operation

```text
   y := alpha*A*x + beta*y,
```

where alpha and beta are scalars, x and y are n element vectors and
A is an n by n symmetric band matrix, with k super-diagonals.
"""

ROUTINE_DOCS["zher"] = """
ZHER   performs the hermitian rank 1 operation

```text
   A := alpha*x*x**H + A,
```

where alpha is a real scalar, x is an n element vector and A is an
n by n hermitian matrix.
"""

ROUTINE_DOCS["srotmg"] = """
CONSTRUCT THE MODIFIED GIVENS TRANSFORMATION MATRIX H WHICH ZEROS
THE SECOND COMPONENT OF THE 2-VECTOR  (SQRT(SD1)*SX1,SQRT(SD2)*>    SY2)**T.
WITH SPARAM(1)=SFLAG, H HAS ONE OF THE FOLLOWING FORMS..

SFLAG=-1.E0     SFLAG=0.E0        SFLAG=1.E0     SFLAG=-2.E0

  (SH11  SH12)    (1.E0  SH12)    (SH11  1.E0)    (1.E0  0.E0)
H=(          )    (          )    (          )    (          )
  (SH21  SH22),   (SH21  1.E0),   (-1.E0 SH22),   (0.E0  1.E0).
LOCATIONS 2-4 OF SPARAM CONTAIN SH11,SH21,SH12, AND SH22
RESPECTIVELY. (VALUES OF 1.E0, -1.E0, OR 0.E0 IMPLIED BY THE
VALUE OF SPARAM(1) ARE NOT STORED IN SPARAM.)

THE VALUES OF GAMSQ AND RGAMSQ SET IN THE DATA STATEMENT MAY BE
INEXACT.  THIS IS OK AS THEY ARE ONLY USED FOR TESTING THE SIZE
OF SD1 AND SD2.  ALL ACTUAL SCALING OF DATA IS DONE USING GAM.
"""

ROUTINE_DOCS["scasum"] = """
SCASUM takes the sum of the (|Re(.)| + |Im(.)|)'s of a complex vector and
returns a single precision result.
"""

ROUTINE_DOCS["dgbmv"] = """
DGBMV  performs one of the matrix-vector operations

```text
   y := alpha*A*x + beta*y,   or   y := alpha*A**T*x + beta*y,
```

where alpha and beta are scalars, x and y are vectors and A is an
m by n band matrix, with kl sub-diagonals and ku super-diagonals.
"""

ROUTINE_DOCS["idamax"] = """
IDAMAX finds the index of the first element having maximum absolute value.
"""

ROUTINE_DOCS["dsymv"] = """
DSYMV  performs the matrix-vector  operation

```text
   y := alpha*A*x + beta*y,
```

where alpha and beta are scalars, x and y are n element vectors and
A is an n by n symmetric matrix.
"""

ROUTINE_DOCS["zhpr"] = """
ZHPR    performs the hermitian rank 1 operation

```text
   A := alpha*x*x**H + A,
```

where alpha is a real scalar, x is an n element vector and A is an
n by n hermitian matrix, supplied in packed form.
"""

ROUTINE_DOCS["zsyr2k"] = """
ZSYR2K  performs one of the symmetric rank 2k operations

```text
   C := alpha*A*B**T + alpha*B*A**T + beta*C,
```

or

```text
   C := alpha*A**T*B + alpha*B**T*A + beta*C,
```

where  alpha and beta  are scalars,  C is an  n by n symmetric matrix
and  A and B  are  n by k  matrices  in the  first  case  and  k by n
matrices in the second case.
"""

ROUTINE_DOCS["zgbmv"] = """
ZGBMV  performs one of the matrix-vector operations

```text
   y := alpha*A*x + beta*y,   or   y := alpha*A**T*x + beta*y,   or
```

```text
   y := alpha*A**H*x + beta*y,
```

where alpha and beta are scalars, x and y are vectors and A is an
m by n band matrix, with kl sub-diagonals and ku super-diagonals.
"""

ROUTINE_DOCS["zherk"] = """
ZHERK  performs one of the hermitian rank k operations

```text
   C := alpha*A*A**H + beta*C,
```

or

```text
   C := alpha*A**H*A + beta*C,
```

where  alpha and beta  are  real scalars,  C is an  n by n  hermitian
matrix and  A  is an  n by k  matrix in the  first case and a  k by n
matrix in the second case.
"""

ROUTINE_DOCS["isamax"] = """
ISAMAX finds the index of the first element having maximum absolute value.
"""

ROUTINE_DOCS["cherk"] = """
CHERK  performs one of the hermitian rank k operations

```text
   C := alpha*A*A**H + beta*C,
```

or

```text
   C := alpha*A**H*A + beta*C,
```

where  alpha and beta  are  real scalars,  C is an  n by n  hermitian
matrix and  A  is an  n by k  matrix in the  first case and a  k by n
matrix in the second case.
"""

ROUTINE_DOCS["ctrmv"] = """
CTRMV  performs one of the matrix-vector operations

```text
   x := A*x,   or   x := A**T*x,   or   x := A**H*x,
```

where x is an n element vector and  A is an n by n unit, or non-unit,
upper or lower triangular matrix.
"""

ROUTINE_DOCS["cher"] = """
CHER   performs the hermitian rank 1 operation

```text
   A := alpha*x*x**H + A,
```

where alpha is a real scalar, x is an n element vector and A is an
n by n hermitian matrix.
"""

ROUTINE_DOCS["scopy"] = """
SCOPY copies a vector, x, to a vector, y.
uses unrolled loops for increments equal to 1.
"""

ROUTINE_DOCS["sasum"] = """
SASUM takes the sum of the absolute values.
uses unrolled loops for increment equal to one.
"""

ROUTINE_DOCS["xerbla_array"] = """
XERBLA_ARRAY assists other languages in calling XERBLA, the LAPACK
and BLAS error handler.  Rather than taking a Fortran string argument
as the function's name, XERBLA_ARRAY takes an array of single
characters along with the array's length.  XERBLA_ARRAY then copies
up to 32 characters of that array into a Fortran string and passes
that to XERBLA.  If called with a non-positive SRNAME_LEN,
XERBLA_ARRAY will call XERBLA with a string of all blank characters.

Say some macro or other device makes XERBLA_ARRAY available to C99
by a name lapack_xerbla and with a common Fortran calling convention.
Then a C99 program could invoke XERBLA via:
```text
   {
```
```text
     int flen = strlen(__func__);
```
```text
     lapack_xerbla(__func__, &flen, &info);
```
```text
   }
```

Providing XERBLA_ARRAY is not necessary for intercepting LAPACK
errors.  XERBLA_ARRAY calls XERBLA.
"""

ROUTINE_DOCS["zswap"] = """
ZSWAP interchanges two vectors.
"""

ROUTINE_DOCS["ctbmv"] = """
CTBMV  performs one of the matrix-vector operations

```text
   x := A*x,   or   x := A**T*x,   or   x := A**H*x,
```

where x is an n element vector and  A is an n by n unit, or non-unit,
upper or lower triangular band matrix, with ( k + 1 ) diagonals.
"""

ROUTINE_DOCS["dcopy"] = """
DCOPY copies a vector, x, to a vector, y.
uses unrolled loops for increments equal to one.
"""

ROUTINE_DOCS["zdotu"] = """
ZDOTU forms the dot product of two complex vectors
```text
     ZDOTU = X^T * Y
```
"""

ROUTINE_DOCS["ztbmv"] = """
ZTBMV  performs one of the matrix-vector operations

```text
   x := A*x,   or   x := A**T*x,   or   x := A**H*x,
```

where x is an n element vector and  A is an n by n unit, or non-unit,
upper or lower triangular band matrix, with ( k + 1 ) diagonals.
"""

ROUTINE_DOCS["ctpsv"] = """
CTPSV  solves one of the systems of equations

```text
   A*x = b,   or   A**T*x = b,   or   A**H*x = b,
```

where b and x are n element vectors and A is an n by n unit, or
non-unit, upper or lower triangular matrix, supplied in packed form.

No test for singularity or near-singularity is included in this
routine. Such tests must be performed before calling this routine.
"""

ROUTINE_DOCS["cswap"] = """
  CSWAP interchanges two vectors.
"""

ROUTINE_DOCS["strmv"] = """
STRMV  performs one of the matrix-vector operations

```text
   x := A*x,   or   x := A**T*x,
```

where x is an n element vector and  A is an n by n unit, or non-unit,
upper or lower triangular matrix.
"""

ROUTINE_DOCS["ztrsv"] = """
ZTRSV  solves one of the systems of equations

```text
   A*x = b,   or   A**T*x = b,   or   A**H*x = b,
```

where b and x are n element vectors and A is an n by n unit, or
non-unit, upper or lower triangular matrix.

No test for singularity or near-singularity is included in this
routine. Such tests must be performed before calling this routine.
"""

ROUTINE_DOCS["icamax"] = """
ICAMAX finds the index of the first element having maximum |Re(.)| + |Im(.)|
"""

ROUTINE_DOCS["dtbsv"] = """
DTBSV  solves one of the systems of equations

```text
   A*x = b,   or   A**T*x = b,
```

where b and x are n element vectors and A is an n by n unit, or
non-unit, upper or lower triangular band matrix, with ( k + 1 )
diagonals.

No test for singularity or near-singularity is included in this
routine. Such tests must be performed before calling this routine.
"""

ROUTINE_DOCS["scnrm2"] = """
SCNRM2 returns the euclidean norm of a vector via the function
name, so that

```text
   SCNRM2 := sqrt( x**H*x )
```
"""

ROUTINE_DOCS["cgbmv"] = """
CGBMV  performs one of the matrix-vector operations

```text
   y := alpha*A*x + beta*y,   or   y := alpha*A**T*x + beta*y,   or
```

```text
   y := alpha*A**H*x + beta*y,
```

where alpha and beta are scalars, x and y are vectors and A is an
m by n band matrix, with kl sub-diagonals and ku super-diagonals.
"""

ROUTINE_DOCS["ssbmv"] = """
SSBMV  performs the matrix-vector  operation

```text
   y := alpha*A*x + beta*y,
```

where alpha and beta are scalars, x and y are n element vectors and
A is an n by n symmetric band matrix, with k super-diagonals.
"""

ROUTINE_DOCS["cdotu"] = """
CDOTU forms the dot product of two complex vectors
```text
     CDOTU = X^T * Y
```
"""

ROUTINE_DOCS["dspr2"] = """
DSPR2  performs the symmetric rank 2 operation

```text
   A := alpha*x*y**T + alpha*y*x**T + A,
```

where alpha is a scalar, x and y are n element vectors and A is an
n by n symmetric matrix, supplied in packed form.
"""

ROUTINE_DOCS["chbmv"] = """
CHBMV  performs the matrix-vector  operation

```text
   y := alpha*A*x + beta*y,
```

where alpha and beta are scalars, x and y are n element vectors and
A is an n by n hermitian band matrix, with k super-diagonals.
"""

ROUTINE_DOCS["zsymm"] = """
ZSYMM  performs one of the matrix-matrix operations

```text
   C := alpha*A*B + beta*C,
```

or

```text
   C := alpha*B*A + beta*C,
```

where  alpha and beta are scalars, A is a symmetric matrix and  B and
C are m by n matrices.
"""

ROUTINE_DOCS["sdsdot"] = """

"""

ROUTINE_DOCS["caxpy"] = """
CAXPY constant times a vector plus a vector.
"""

ROUTINE_DOCS["ztrmm"] = """
ZTRMM  performs one of the matrix-matrix operations

```text
   B := alpha*op( A )*B,   or   B := alpha*B*op( A )
```

where  alpha  is a scalar,  B  is an m by n matrix,  A  is a unit, or
non-unit,  upper or lower triangular matrix  and  op( A )  is one  of

```text
   op( A ) = A   or   op( A ) = A**T   or   op( A ) = A**H.
```
"""

ROUTINE_DOCS["cdotc"] = """
CDOTC forms the dot product of two complex vectors
```text
     CDOTC = X^H * Y
```
"""

ROUTINE_DOCS["dtpmv"] = """
DTPMV  performs one of the matrix-vector operations

```text
   x := A*x,   or   x := A**T*x,
```

where x is an n element vector and  A is an n by n unit, or non-unit,
upper or lower triangular matrix, supplied in packed form.
"""

ROUTINE_DOCS["dgemm"] = """
DGEMM  performs one of the matrix-matrix operations

```text
   C := alpha*op( A )*op( B ) + beta*C,
```

where  op( X ) is one of

```text
   op( X ) = X   or   op( X ) = X**T,
```

alpha and beta are scalars, and A, B and C are matrices, with op( A )
an m by k matrix,  op( B )  a  k by n matrix and  C an m by n matrix.
"""

ROUTINE_DOCS["zher2k"] = """
ZHER2K  performs one of the hermitian rank 2k operations

```text
   C := alpha*A*B**H + conjg( alpha )*B*A**H + beta*C,
```

or

```text
   C := alpha*A**H*B + conjg( alpha )*B**H*A + beta*C,
```

where  alpha and beta  are scalars with  beta  real,  C is an  n by n
hermitian matrix and  A and B  are  n by k matrices in the first case
and  k by n  matrices in the second case.
"""

ROUTINE_DOCS["stpmv"] = """
STPMV  performs one of the matrix-vector operations

```text
   x := A*x,   or   x := A**T*x,
```

where x is an n element vector and  A is an n by n unit, or non-unit,
upper or lower triangular matrix, supplied in packed form.
"""

ROUTINE_DOCS["zaxpy"] = """
ZAXPY constant times a vector plus a vector.
"""

ROUTINE_DOCS["dtrmm"] = """
DTRMM  performs one of the matrix-matrix operations

```text
   B := alpha*op( A )*B,   or   B := alpha*B*op( A ),
```

where  alpha  is a scalar,  B  is an m by n matrix,  A  is a unit, or
non-unit,  upper or lower triangular matrix  and  op( A )  is one  of

```text
   op( A ) = A   or   op( A ) = A**T.
```
"""

ROUTINE_DOCS["ctpmv"] = """
CTPMV  performs one of the matrix-vector operations

```text
   x := A*x,   or   x := A**T*x,   or   x := A**H*x,
```

where x is an n element vector and  A is an n by n unit, or non-unit,
upper or lower triangular matrix, supplied in packed form.
"""

ROUTINE_DOCS["cher2"] = """
CHER2  performs the hermitian rank 2 operation

```text
   A := alpha*x*y**H + conjg( alpha )*y*x**H + A,
```

where alpha is a scalar, x and y are n element vectors and A is an n
by n hermitian matrix.
"""

ROUTINE_DOCS["dznrm2"] = """
DZNRM2 returns the euclidean norm of a vector via the function
name, so that

```text
   DZNRM2 := sqrt( x**H*x )
```
"""

ROUTINE_DOCS["ccopy"] = """
CCOPY copies a vector x to a vector y.
"""

ROUTINE_DOCS["dspr"] = """
DSPR    performs the symmetric rank 1 operation

```text
   A := alpha*x*x**T + A,
```

where alpha is a real scalar, x is an n element vector and A is an
n by n symmetric matrix, supplied in packed form.
"""

ROUTINE_DOCS["dsymm"] = """
DSYMM  performs one of the matrix-matrix operations

```text
   C := alpha*A*B + beta*C,
```

or

```text
   C := alpha*B*A + beta*C,
```

where alpha and beta are scalars,  A is a symmetric matrix and  B and
C are  m by n matrices.
"""

ROUTINE_DOCS["daxpy"] = """
DAXPY constant times a vector plus a vector.
uses unrolled loops for increments equal to one.
"""

ROUTINE_DOCS["chemv"] = """
CHEMV  performs the matrix-vector  operation

```text
   y := alpha*A*x + beta*y,
```

where alpha and beta are scalars, x and y are n element vectors and
A is an n by n hermitian matrix.
"""

ROUTINE_DOCS["drotm"] = """
APPLY THE MODIFIED GIVENS TRANSFORMATION, H, TO THE 2 BY N MATRIX

(DX**T) , WHERE **T INDICATES TRANSPOSE. THE ELEMENTS OF DX ARE IN
(DY**T)

DX(LX+I*INCX), I = 0 TO N-1, WHERE LX = 1 IF INCX .GE. 0, ELSE
LX = (-INCX)*N, AND SIMILARLY FOR SY USING LY AND INCY.
WITH DPARAM(1)=DFLAG, H HAS ONE OF THE FOLLOWING FORMS..

DFLAG=-1.D0     DFLAG=0.D0        DFLAG=1.D0     DFLAG=-2.D0

  (DH11  DH12)    (1.D0  DH12)    (DH11  1.D0)    (1.D0  0.D0)
H=(          )    (          )    (          )    (          )
  (DH21  DH22),   (DH21  1.D0),   (-1.D0 DH22),   (0.D0  1.D0).
SEE DROTMG FOR A DESCRIPTION OF DATA STORAGE IN DPARAM.
"""

ROUTINE_DOCS["zsyrk"] = """
ZSYRK  performs one of the symmetric rank k operations

```text
   C := alpha*A*A**T + beta*C,
```

or

```text
   C := alpha*A**T*A + beta*C,
```

where  alpha and beta  are scalars,  C is an  n by n symmetric matrix
and  A  is an  n by k  matrix in the first case and a  k by n  matrix
in the second case.
"""

ROUTINE_DOCS["zher2"] = """
ZHER2  performs the hermitian rank 2 operation

```text
   A := alpha*x*y**H + conjg( alpha )*y*x**H + A,
```

where alpha is a scalar, x and y are n element vectors and A is an n
by n hermitian matrix.
"""

ROUTINE_DOCS["strmm"] = """
STRMM  performs one of the matrix-matrix operations

```text
   B := alpha*op( A )*B,   or   B := alpha*B*op( A ),
```

where  alpha  is a scalar,  B  is an m by n matrix,  A  is a unit, or
non-unit,  upper or lower triangular matrix  and  op( A )  is one  of

```text
   op( A ) = A   or   op( A ) = A**T.
```
"""

ROUTINE_DOCS["zscal"] = """
ZSCAL scales a vector by a constant.
"""

ROUTINE_DOCS["chpr"] = """
CHPR    performs the hermitian rank 1 operation

```text
   A := alpha*x*x**H + A,
```

where alpha is a real scalar, x is an n element vector and A is an
n by n hermitian matrix, supplied in packed form.
"""

ROUTINE_DOCS["dsyr2k"] = """
DSYR2K  performs one of the symmetric rank 2k operations

```text
   C := alpha*A*B**T + alpha*B*A**T + beta*C,
```

or

```text
   C := alpha*A**T*B + alpha*B**T*A + beta*C,
```

where  alpha and beta  are scalars, C is an  n by n  symmetric matrix
and  A and B  are  n by k  matrices  in the  first  case  and  k by n
matrices in the second case.
"""

ROUTINE_DOCS["lsame"] = """
LSAME returns .TRUE. if CA is the same letter as CB regardless of
case.
"""

ROUTINE_DOCS["zdotc"] = """
ZDOTC forms the dot product of two complex vectors
```text
     ZDOTC = X^H * Y
```
"""

ROUTINE_DOCS["ctrmm"] = """
CTRMM  performs one of the matrix-matrix operations

```text
   B := alpha*op( A )*B,   or   B := alpha*B*op( A )
```

where  alpha  is a scalar,  B  is an m by n matrix,  A  is a unit, or
non-unit,  upper or lower triangular matrix  and  op( A )  is one  of

```text
   op( A ) = A   or   op( A ) = A**T   or   op( A ) = A**H.
```
"""

ROUTINE_DOCS["ssymv"] = """
SSYMV  performs the matrix-vector  operation

```text
   y := alpha*A*x + beta*y,
```

where alpha and beta are scalars, x and y are n element vectors and
A is an n by n symmetric matrix.
"""

ROUTINE_DOCS["ztrsm"] = """
ZTRSM  solves one of the matrix equations

```text
   op( A )*X = alpha*B,   or   X*op( A ) = alpha*B,
```

where alpha is a scalar, X and B are m by n matrices, A is a unit, or
non-unit,  upper or lower triangular matrix  and  op( A )  is one  of

```text
   op( A ) = A   or   op( A ) = A**T   or   op( A ) = A**H.
```

The matrix X is overwritten on B.
"""

ROUTINE_DOCS["sspr2"] = """
SSPR2  performs the symmetric rank 2 operation

```text
   A := alpha*x*y**T + alpha*y*x**T + A,
```

where alpha is a scalar, x and y are n element vectors and A is an
n by n symmetric matrix, supplied in packed form.
"""

ROUTINE_DOCS["stpsv"] = """
STPSV  solves one of the systems of equations

```text
   A*x = b,   or   A**T*x = b,
```

where b and x are n element vectors and A is an n by n unit, or
non-unit, upper or lower triangular matrix, supplied in packed form.

No test for singularity or near-singularity is included in this
routine. Such tests must be performed before calling this routine.
"""

ROUTINE_DOCS["strsm"] = """
STRSM  solves one of the matrix equations

```text
   op( A )*X = alpha*B,   or   X*op( A ) = alpha*B,
```

where alpha is a scalar, X and B are m by n matrices, A is a unit, or
non-unit,  upper or lower triangular matrix  and  op( A )  is one  of

```text
   op( A ) = A   or   op( A ) = A**T.
```

The matrix X is overwritten on B.
"""

ROUTINE_DOCS["zgemv"] = """
ZGEMV  performs one of the matrix-vector operations

```text
   y := alpha*A*x + beta*y,   or   y := alpha*A**T*x + beta*y,   or
```

```text
   y := alpha*A**H*x + beta*y,
```

where alpha and beta are scalars, x and y are vectors and A is an
m by n matrix.
"""

ROUTINE_DOCS["sgbmv"] = """
SGBMV  performs one of the matrix-vector operations

```text
   y := alpha*A*x + beta*y,   or   y := alpha*A**T*x + beta*y,
```

where alpha and beta are scalars, x and y are vectors and A is an
m by n band matrix, with kl sub-diagonals and ku super-diagonals.
"""

ROUTINE_DOCS["dspmv"] = """
DSPMV  performs the matrix-vector operation

```text
   y := alpha*A*x + beta*y,
```

where alpha and beta are scalars, x and y are n element vectors and
A is an n by n symmetric matrix, supplied in packed form.
"""

ROUTINE_DOCS["dtpsv"] = """
DTPSV  solves one of the systems of equations

```text
   A*x = b,   or   A**T*x = b,
```

where b and x are n element vectors and A is an n by n unit, or
non-unit, upper or lower triangular matrix, supplied in packed form.

No test for singularity or near-singularity is included in this
routine. Such tests must be performed before calling this routine.
"""

ROUTINE_DOCS["crotg"] = """
CROTG determines a complex Givens rotation.
"""

ROUTINE_DOCS["saxpy"] = """
SAXPY constant times a vector plus a vector.
uses unrolled loops for increments equal to one.
"""

ROUTINE_DOCS["drot"] = """
DROT applies a plane rotation.
"""

ROUTINE_DOCS["cgeru"] = """
CGERU  performs the rank 1 operation

```text
   A := alpha*x*y**T + A,
```

where alpha is a scalar, x is an m element vector, y is an n element
vector and A is an m by n matrix.
"""

ROUTINE_DOCS["dtbmv"] = """
DTBMV  performs one of the matrix-vector operations

```text
   x := A*x,   or   x := A**T*x,
```

where x is an n element vector and  A is an n by n unit, or non-unit,
upper or lower triangular band matrix, with ( k + 1 ) diagonals.
"""

ROUTINE_DOCS["ctrsm"] = """
CTRSM  solves one of the matrix equations

```text
   op( A )*X = alpha*B,   or   X*op( A ) = alpha*B,
```

where alpha is a scalar, X and B are m by n matrices, A is a unit, or
non-unit,  upper or lower triangular matrix  and  op( A )  is one  of

```text
   op( A ) = A   or   op( A ) = A**T   or   op( A ) = A**H.
```

The matrix X is overwritten on B.
"""

ROUTINE_DOCS["drotg"] = """
DROTG construct givens plane rotation.
"""

ROUTINE_DOCS["chpr2"] = """
CHPR2  performs the hermitian rank 2 operation

```text
   A := alpha*x*y**H + conjg( alpha )*y*x**H + A,
```

where alpha is a scalar, x and y are n element vectors and A is an
n by n hermitian matrix, supplied in packed form.
"""

ROUTINE_DOCS["zdrot"] = """
Applies a plane rotation, where the cos and sin (c and s) are real
and the vectors cx and cy are complex.
jack dongarra, linpack, 3/11/78.
"""

ROUTINE_DOCS["sger"] = """
SGER   performs the rank 1 operation

```text
   A := alpha*x*y**T + A,
```

where alpha is a scalar, x is an m element vector, y is an n element
vector and A is an m by n matrix.
"""

ROUTINE_DOCS["cher2k"] = """
CHER2K  performs one of the hermitian rank 2k operations

```text
   C := alpha*A*B**H + conjg( alpha )*B*A**H + beta*C,
```

or

```text
   C := alpha*A**H*B + conjg( alpha )*B**H*A + beta*C,
```

where  alpha and beta  are scalars with  beta  real,  C is an  n by n
hermitian matrix and  A and B  are  n by k matrices in the first case
and  k by n  matrices in the second case.
"""

ROUTINE_DOCS["chemm"] = """
CHEMM  performs one of the matrix-matrix operations

```text
   C := alpha*A*B + beta*C,
```

or

```text
   C := alpha*B*A + beta*C,
```

where alpha and beta are scalars, A is an hermitian matrix and  B and
C are m by n matrices.
"""

ROUTINE_DOCS["ztrmv"] = """
ZTRMV  performs one of the matrix-vector operations

```text
   x := A*x,   or   x := A**T*x,   or   x := A**H*x,
```

where x is an n element vector and  A is an n by n unit, or non-unit,
upper or lower triangular matrix.
"""

ROUTINE_DOCS["dswap"] = """
interchanges two vectors.
uses unrolled loops for increments equal one.
"""

ROUTINE_DOCS["chpmv"] = """
CHPMV  performs the matrix-vector operation

```text
   y := alpha*A*x + beta*y,
```

where alpha and beta are scalars, x and y are n element vectors and
A is an n by n hermitian matrix, supplied in packed form.
"""

ROUTINE_DOCS["ssymm"] = """
SSYMM  performs one of the matrix-matrix operations

```text
   C := alpha*A*B + beta*C,
```

or

```text
   C := alpha*B*A + beta*C,
```

where alpha and beta are scalars,  A is a symmetric matrix and  B and
C are  m by n matrices.
"""

ROUTINE_DOCS["dcabs1"] = """
DCABS1 computes |Re(.)| + |Im(.)| of a double complex number
"""

ROUTINE_DOCS["zgeru"] = """
ZGERU  performs the rank 1 operation

```text
   A := alpha*x*y**T + A,
```

where alpha is a scalar, x is an m element vector, y is an n element
vector and A is an m by n matrix.
"""

ROUTINE_DOCS["srotm"] = """
APPLY THE MODIFIED GIVENS TRANSFORMATION, H, TO THE 2 BY N MATRIX

(SX**T) , WHERE **T INDICATES TRANSPOSE. THE ELEMENTS OF SX ARE IN
(SX**T)

SX(LX+I*INCX), I = 0 TO N-1, WHERE LX = 1 IF INCX .GE. 0, ELSE
LX = (-INCX)*N, AND SIMILARLY FOR SY USING USING LY AND INCY.
WITH SPARAM(1)=SFLAG, H HAS ONE OF THE FOLLOWING FORMS..

SFLAG=-1.E0     SFLAG=0.E0        SFLAG=1.E0     SFLAG=-2.E0

  (SH11  SH12)    (1.E0  SH12)    (SH11  1.E0)    (1.E0  0.E0)
H=(          )    (          )    (          )    (          )
  (SH21  SH22),   (SH21  1.E0),   (-1.E0 SH22),   (0.E0  1.E0).
SEE  SROTMG FOR A DESCRIPTION OF DATA STORAGE IN SPARAM.
"""

ROUTINE_DOCS["csymm"] = """
CSYMM  performs one of the matrix-matrix operations

```text
   C := alpha*A*B + beta*C,
```

or

```text
   C := alpha*B*A + beta*C,
```

where  alpha and beta are scalars, A is a symmetric matrix and  B and
C are m by n matrices.
"""

ROUTINE_DOCS["csyr2k"] = """
CSYR2K  performs one of the symmetric rank 2k operations

```text
   C := alpha*A*B**T + alpha*B*A**T + beta*C,
```

or

```text
   C := alpha*A**T*B + alpha*B**T*A + beta*C,
```

where  alpha and beta  are scalars,  C is an  n by n symmetric matrix
and  A and B  are  n by k  matrices  in the  first  case  and  k by n
matrices in the second case.
"""

ROUTINE_DOCS["csyrk"] = """
CSYRK  performs one of the symmetric rank k operations

```text
   C := alpha*A*A**T + beta*C,
```

or

```text
   C := alpha*A**T*A + beta*C,
```

where  alpha and beta  are scalars,  C is an  n by n symmetric matrix
and  A  is an  n by k  matrix in the first case and a  k by n  matrix
in the second case.
"""

ROUTINE_DOCS["scabs1"] = """
SCABS1 computes |Re(.)| + |Im(.)| of a complex number
"""

ROUTINE_DOCS["izamax"] = """
IZAMAX finds the index of the first element having maximum |Re(.)| + |Im(.)|
"""

ROUTINE_DOCS["dtrmv"] = """
DTRMV  performs one of the matrix-vector operations

```text
   x := A*x,   or   x := A**T*x,
```

where x is an n element vector and  A is an n by n unit, or non-unit,
upper or lower triangular matrix.
"""

ROUTINE_DOCS["dscal"] = """
DSCAL scales a vector by a constant.
uses unrolled loops for increment equal to one.
"""

ROUTINE_DOCS["dsyrk"] = """
DSYRK  performs one of the symmetric rank k operations

```text
   C := alpha*A*A**T + beta*C,
```

or

```text
   C := alpha*A**T*A + beta*C,
```

where  alpha and beta  are scalars, C is an  n by n  symmetric matrix
and  A  is an  n by k  matrix in the first case and a  k by n  matrix
in the second case.
"""

ROUTINE_DOCS["ztpsv"] = """
ZTPSV  solves one of the systems of equations

```text
   A*x = b,   or   A**T*x = b,   or   A**H*x = b,
```

where b and x are n element vectors and A is an n by n unit, or
non-unit, upper or lower triangular matrix, supplied in packed form.

No test for singularity or near-singularity is included in this
routine. Such tests must be performed before calling this routine.
"""

ROUTINE_DOCS["cgerc"] = """
CGERC  performs the rank 1 operation

```text
   A := alpha*x*y**H + A,
```

where alpha is a scalar, x is an m element vector, y is an n element
vector and A is an m by n matrix.
"""

ROUTINE_DOCS["stbsv"] = """
STBSV  solves one of the systems of equations

```text
   A*x = b,   or   A**T*x = b,
```

where b and x are n element vectors and A is an n by n unit, or
non-unit, upper or lower triangular band matrix, with ( k + 1 )
diagonals.

No test for singularity or near-singularity is included in this
routine. Such tests must be performed before calling this routine.
"""

ROUTINE_DOCS["sspr"] = """
SSPR    performs the symmetric rank 1 operation

```text
   A := alpha*x*x**T + A,
```

where alpha is a real scalar, x is an n element vector and A is an
n by n symmetric matrix, supplied in packed form.
"""

ROUTINE_DOCS["ssyrk"] = """
SSYRK  performs one of the symmetric rank k operations

```text
   C := alpha*A*A**T + beta*C,
```

or

```text
   C := alpha*A**T*A + beta*C,
```

where  alpha and beta  are scalars, C is an  n by n  symmetric matrix
and  A  is an  n by k  matrix in the first case and a  k by n  matrix
in the second case.
"""

ROUTINE_DOCS["zhemv"] = """
ZHEMV  performs the matrix-vector  operation

```text
   y := alpha*A*x + beta*y,
```

where alpha and beta are scalars, x and y are n element vectors and
A is an n by n hermitian matrix.
"""

ROUTINE_DOCS["dtrsv"] = """
DTRSV  solves one of the systems of equations

```text
   A*x = b,   or   A**T*x = b,
```

where b and x are n element vectors and A is an n by n unit, or
non-unit, upper or lower triangular matrix.

No test for singularity or near-singularity is included in this
routine. Such tests must be performed before calling this routine.
"""

ROUTINE_DOCS["dasum"] = """
DASUM takes the sum of the absolute values.
"""

ROUTINE_DOCS["dger"] = """
DGER   performs the rank 1 operation

```text
   A := alpha*x*y**T + A,
```

where alpha is a scalar, x is an m element vector, y is an n element
vector and A is an m by n matrix.
"""

ROUTINE_DOCS["xerbla"] = """
XERBLA  is an error handler for the LAPACK routines.
It is called by an LAPACK routine if an input parameter has an
invalid value.  A message is printed and execution stops.

Installers may consider modifying the STOP statement in order to
call system-specific exception-handling facilities.
"""

ROUTINE_DOCS["ztbsv"] = """
ZTBSV  solves one of the systems of equations

```text
   A*x = b,   or   A**T*x = b,   or   A**H*x = b,
```

where b and x are n element vectors and A is an n by n unit, or
non-unit, upper or lower triangular band matrix, with ( k + 1 )
diagonals.

No test for singularity or near-singularity is included in this
routine. Such tests must be performed before calling this routine.
"""

ROUTINE_DOCS["ctbsv"] = """
CTBSV  solves one of the systems of equations

```text
   A*x = b,   or   A**T*x = b,   or   A**H*x = b,
```

where b and x are n element vectors and A is an n by n unit, or
non-unit, upper or lower triangular band matrix, with ( k + 1 )
diagonals.

No test for singularity or near-singularity is included in this
routine. Such tests must be performed before calling this routine.
"""

ROUTINE_DOCS["zhbmv"] = """
ZHBMV  performs the matrix-vector  operation

```text
   y := alpha*A*x + beta*y,
```

where alpha and beta are scalars, x and y are n element vectors and
A is an n by n hermitian band matrix, with k super-diagonals.
"""

ROUTINE_DOCS["dsyr2"] = """
DSYR2  performs the symmetric rank 2 operation

```text
   A := alpha*x*y**T + alpha*y*x**T + A,
```

where alpha is a scalar, x and y are n element vectors and A is an n
by n symmetric matrix.
"""

ROUTINE_DOCS["sspmv"] = """
SSPMV  performs the matrix-vector operation

```text
   y := alpha*A*x + beta*y,
```

where alpha and beta are scalars, x and y are n element vectors and
A is an n by n symmetric matrix, supplied in packed form.
"""

ROUTINE_DOCS["zrotg"] = """
ZROTG determines a double complex Givens rotation.
"""

ROUTINE_DOCS["dsyr"] = """
DSYR   performs the symmetric rank 1 operation

```text
   A := alpha*x*x**T + A,
```

where alpha is a real scalar, x is an n element vector and A is an
n by n symmetric matrix.
"""

ROUTINE_DOCS["strsv"] = """
STRSV  solves one of the systems of equations

```text
   A*x = b,   or   A**T*x = b,
```

where b and x are n element vectors and A is an n by n unit, or
non-unit, upper or lower triangular matrix.

No test for singularity or near-singularity is included in this
routine. Such tests must be performed before calling this routine.
"""

ROUTINE_DOCS["ssyr2"] = """
SSYR2  performs the symmetric rank 2 operation

```text
   A := alpha*x*y**T + alpha*y*x**T + A,
```

where alpha is a scalar, x and y are n element vectors and A is an n
by n symmetric matrix.
"""

ROUTINE_DOCS["ddot"] = """
DDOT forms the dot product of two vectors.
uses unrolled loops for increments equal to one.
"""

ROUTINE_DOCS["dnrm2"] = """
DNRM2 returns the euclidean norm of a vector via the function
name, so that

```text
   DNRM2 := sqrt( x'*x )
```
"""

