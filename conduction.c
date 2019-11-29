#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mpi.h"
#ifndef W
#define W 20                                    // Width
#define FROM_LEFT 0
#define FROM_RIGHT 1
#endif

int main(int argc, char **argv) {
  int rank, size;
  MPI_Init(&argc, &argv);
  MPI_Status status;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int L = atoi(argv[1]);                        // Length
  int iteration = atoi(argv[2]);                // Iteration
  srand(atoi(argv[3]));                         // Seed
  float d = (float) random() / RAND_MAX * 0.2;  // Diffusivity
  int *temp = malloc(L*W*sizeof(int));          // Current temperature
  int *next = malloc(L*W*sizeof(int));          // Next time step

  for (int i = 0; i < L; i++) {
    for (int j = 0; j < W; j++) {
      temp[i*W+j] = random()>>3;
    }
  }

  int avg_rows, extra;
  int offset, rows;
  avg_rows = L/size;
  extra = L%size;
  rows = (rank < extra) ? (avg_rows + 1) : avg_rows;
  offset = (rank * avg_rows) + ((rank < extra) ? rank : extra);

  int left[W], right[W];
  memcpy(&left, (rank==0) ? temp : (temp+((offset-1)*W)), sizeof(int)*W);
  memcpy(&right, (rank+1==size) ? temp+((L-1)*W) : (temp+(offset+rows)*W), sizeof(int)*W);
  // DEBUG
  /*
  if (rank==3) {
    for (int i=0; i<W; i++) {
      for (int j=0; j<L; j++) {
        printf("%d ", temp[j*W+i]);
      }
      printf("\n");
    }

    printf("\n\n\n\n");

    for (int i=0; i<W; i++) {
      printf("%d %d\n", left[i], right[i]);
    }
  }
  */
  int count = 0, local_balance = 0, global_balance;
  while (iteration--) {     // Compute with up, left, right, down points
    local_balance = 1;
    count++;
    for (int i = offset; i < offset+rows; i++) {
      for (int j = 0; j < W; j++) {
        float t = temp[i*W+j] / d;
        t += temp[i*W+j] * -4;
        t += (i==offset) ? left[j] : temp[(i-1)*W+j];
        t += (i+1==offset+rows) ? right[j] : temp[(i+1)*W+j];
        t += temp[i*W+(j - 1 <  0 ? 0 : j - 1)];
        t += temp[i*W+(j + 1 >= W ? j : j + 1)];
        t *= d;
        next[i*W+j] = t ;
        if (next[i*W+j] != temp[i*W+j]) {
          local_balance = 0;
        }
      }
    }

    MPI_Allreduce(&local_balance, &global_balance, 1, MPI_INT, MPI_BAND, MPI_COMM_WORLD);
    if (global_balance) {
      break;
    }
    int *tmp = temp;
    temp = next;
    next = tmp;

    if (rank!=0) {
      MPI_Send(temp+(offset*W), W, MPI_INT, rank-1, FROM_RIGHT, MPI_COMM_WORLD);
    }
    if (rank+1!=size) {
      MPI_Send(temp+((offset+rows-1)*W), W, MPI_INT, rank+1, FROM_LEFT, MPI_COMM_WORLD);
    }

    if (rank!=0) {
      MPI_Recv(&left, W, MPI_INT, rank-1, FROM_LEFT, MPI_COMM_WORLD, &status);
    } else {
      memcpy(&left, (rank==0) ? temp : (temp+((offset-2)*W)), sizeof(int)*W);
    }
    if (rank+1!=size) {
      MPI_Recv(&right, W, MPI_INT, rank+1, FROM_RIGHT, MPI_COMM_WORLD, &status);
    } else {
      memcpy(&right, (rank+1==size) ? temp+((L-1)*W) : (temp+(offset+rows-1)*W), sizeof(int)*W);      
    }
  }
  int local_min = temp[offset*W];
  for (int i = offset; i < offset+rows; i++) {
    for (int j = 0; j < W; j++) {
      if (temp[i*W+j] < local_min) {
        local_min = temp[i*W+j];
      }
    }
  }
  int global_min;
  MPI_Reduce(&local_min, &global_min, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
  if (rank==0) {
    printf("Size: %d*%d, Iteration: %d, Min Temp: %d\n", L, W, count, global_min);
  }

  MPI_Finalize();
  return 0;
}