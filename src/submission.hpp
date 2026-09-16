#pragma once
#include <cstddef>
#include <vector>
#include <omp.h>

// Starter Grid for the 2D heat-diffusion problem.
//
// The evaluation harness uses operator() to set initial conditions and to read
// results; it never touches your internal storage. Keep this interface,
// everything else is yours.
class Grid {
private:
  std::size_t rows_;
  std::size_t cols_;

  // could use a uniqueptr or something but this is fine for now
  std::vector<double> data_;

  // just for simplication
  // CHECK later to ensure this is row major order but I think this is right
  std::size_t index(std::size_t row, std::size_t col) const {
    return row * cols_ + col;
  }

public:
  Grid(std::size_t rows, std::size_t cols) : rows_(rows), cols_(cols), data_(rows * cols, 0.0) {};

  double& operator()(std::size_t i, std::size_t j) {
    return data_[index(i, j)];
  };

  double operator()(std::size_t i, std::size_t j) const {
    return data_[index(i, j)];
  }

  // start with exposing some methods for getting rows/cols
  std::size_t get_row_count() const { return rows_; }
  std::size_t get_col_count() const { return cols_; }

};  

// Apply the five-point stencil over all interior points, copying the boundary
// values unchanged from old_grid to new_grid. Implement your solution here.
void apply_stencil(const Grid& old_grid, Grid& new_grid) {

  std::size_t rows = new_grid.get_row_count();
  std::size_t cols = new_grid.get_col_count();

  for(std::size_t j = 0; j < cols; ++j) {
    new_grid(0,j) = old_grid(0,j);
  }

  for(std::size_t j = 0; j < cols; ++j) {
    new_grid(rows - 1,j) = old_grid(rows - 1,j);
  }
  
  for(std::size_t i = 0; i < rows; ++i) {
    new_grid(i,0) = old_grid(i,0);
  }

  for(std::size_t i = 0; i < rows; ++i) {
    new_grid(i,cols - 1) = old_grid(i,cols - 1);
  }

  // collpase did not help
  #pragma omp parallel for schedule(static)
  for (size_t i = 1; i < rows - 1; i++) {
    #pragma omp simd

    for(size_t j = 1; j < cols - 1; j++) {
      // these are the exterior points to skip
      // actually we can just change the loop condition
      // if (i == 0 or i == rows - 1 or j == 0 or j == cols - 1) {
      //   new_grid(i,j) = old_grid(i,j);
      //   continue;
      // }
      new_grid(i, j) = 0.5 * old_grid(i,j) + 0.125 * (old_grid(i-1,j) + old_grid(i+1,j) + old_grid(i,j-1) + old_grid(i,j+1));
    }
  }
}