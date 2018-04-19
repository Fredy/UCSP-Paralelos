#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <mpi.h>

using namespace std;

int *createMatrix(int rows, int cols) {
  int *matrix = new int[rows * cols];
  fill_n(matrix, rows * cols, 1);
  /*
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      matrix[cols * i + j] = 1;
    }
  }
  */
  return matrix;
}

int *createVector(int rows) {
  int *vector = new int[rows];
  fill_n(vector, rows, 1);
  return vector;
}

int main(int argc, char *argv[]) {
  int rank, numprocs, rows, cols, count, remainder, myRowsSize;
  int *matrix = nullptr;
  int *vector = nullptr;
  int *result = nullptr;
  int *sendcounts = nullptr;
  int *senddispls = nullptr;
  int *recvcounts = nullptr;
  int *recvdispls = nullptr;
  double start, finish;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  start = MPI_Wtime();

  if (0 == rank) {
    /*
    printf("Product of a vector by a matrix\n");
    printf("Enter the number of matrix rows:");
    scanf("%i", &rows);

    if (rows < 1) {
      return EXIT_FAILURE;
    }

    printf("Enter the number of matrix columns:");
    scanf("%i", &cols);

    if (cols < 1) {
      return EXIT_FAILURE;
    }
    */
    cols = 50000;
    rows = 10000;
    matrix = createMatrix(rows, cols);
    vector = createVector(cols);

    sendcounts = new int[numprocs];
    senddispls = new int[numprocs];
    recvcounts = new int[numprocs];
    recvdispls = new int[numprocs];

    count = rows / numprocs;
    remainder = rows - count * numprocs;
    int prefixSum = 0;
    for (int i = 0; i < numprocs; ++i) {
      recvcounts[i] = (i < remainder) ? count + 1 : count;
      sendcounts[i] = recvcounts[i] * cols;
      recvdispls[i] = prefixSum;
      senddispls[i] = prefixSum * cols;
      prefixSum += recvcounts[i];
    }
  }

  MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (0 != rank) {
    vector = new int[cols];
  }

  MPI_Bcast(vector, cols, MPI_INT, 0, MPI_COMM_WORLD);

  if (0 != rank) {
    count = rows / numprocs;
    remainder = rows - count * numprocs;
  }

  myRowsSize = rank < remainder ? count + 1 : count;
  int *matrixPart = new int[myRowsSize * cols];

  MPI_Scatterv(matrix, sendcounts, senddispls, MPI_INT, matrixPart,
               myRowsSize * cols, MPI_INT, 0, MPI_COMM_WORLD);

  if (0 == rank) {
    delete[] sendcounts;
    delete[] senddispls;
    delete[] matrix;
  }

  int *resultPart = new int[myRowsSize];

#pragma omp parallel for
  for (int i = 0; i < myRowsSize; ++i) {
    resultPart[i] = 0;
    for (int j = 0; j < cols; ++j) {
      resultPart[i] += matrixPart[i * cols + j] * vector[j];
    }
  }

  delete[] matrixPart;
  delete[] vector;

  if (0 == rank) {
    result = new int[rows];
  }

  MPI_Gatherv(resultPart, myRowsSize, MPI_INT, result, recvcounts, recvdispls,
              MPI_INT, 0, MPI_COMM_WORLD);

  delete[] resultPart;

  if (0 == rank) {
    delete[] recvcounts;
    delete[] recvdispls;
  }

  finish = MPI_Wtime();
  cout << "PROC: " << rank << "Elapsd time = " << finish - start
       << " seconds.\n";

  MPI_Finalize();

  if (0 == rank) {
    /*
    printf("result: \n");
    for (int i = 0; i < rows; ++i)
      printf("%i ", result[i]);

    printf("\n");
    */
    delete[] result;
  }
}
