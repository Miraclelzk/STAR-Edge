#ifndef _COSPML_H
#define _COSPML_H

// TODO rename

int TableSize(const int, const int);

int Spharmonic_TableSize(const int);

int Reduced_SpharmonicTableSize(const int, const int);

int Reduced_Naive_TableSize(const int, const int);

int TableOffset(int, int);

void GenerateCosPmlTable(const int, const int, double*, double*);

int RowSize(const int, const int);

int Transpose_RowSize(const int, const int, const int);

void TransposeCosPmlTable(const int, const int, double*, double*);

double** Spharmonic_Pml_Table(const int, double*, double*);

double** Transpose_Spharmonic_Pml_Table(double**, const int, double*);

double** SemiNaive_Naive_Pml_Table(const int, const int, double*, double*);

double** Transpose_SemiNaive_Naive_Pml_Table(double**, const int, const int, double*, double*);

#endif // _COSPML_H
