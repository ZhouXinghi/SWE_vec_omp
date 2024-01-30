/**
 * @file
 * This file is part of SWE.
 *
 * @author Alexander Breuer (breuera AT in.tum.de, http://www5.in.tum.de/wiki/index.php/Dipl.-Math._Alexander_Breuer)
 * @author Sebastian Rettenberger (rettenbs AT in.tum.de,
 * http://www5.in.tum.de/wiki/index.php/Sebastian_Rettenberger,_M.Sc.)
 * @author Michael Bader (bader AT in.tum.de, http://www5.in.tum.de/wiki/index.php/Michael_Bader)
 *
 * @section LICENSE
 *
 * SWE is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * SWE is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with SWE.  If not, see <http://www.gnu.org/licenses/>.
 *
 *
 * @section DESCRIPTION
 *
 * Implementation of Blocks::Block that uses solvers in the wave propagation formulation.
 */

#include "WavePropagationBlock.hpp"
#include "../Tools/Logger.hpp"
#include <iostream>
#include <omp.h>

Blocks::WavePropagationBlock::WavePropagationBlock(int nx, int ny, RealType dx, RealType dy):
  Block(nx, ny, dx, dy),
  hNetUpdatesLeft_(nx + 1, ny),
  hNetUpdatesRight_(nx + 1, ny),
  huNetUpdatesLeft_(nx + 1, ny),
  huNetUpdatesRight_(nx + 1, ny),
  hNetUpdatesBelow_(nx, ny + 1),
  hNetUpdatesAbove_(nx, ny + 1),
  hvNetUpdatesBelow_(nx, ny + 1),
  hvNetUpdatesAbove_(nx, ny + 1),
  wavePropagationSolver_(omp_get_max_threads()),
  wavePropagationSolversSIMD_(omp_get_max_threads())
  {
    // for (size_t i = 0; i < omp_get_max_threads(); i++)
    // {
    //   wavePropagationSolver_.push_back(Solvers::FWaveSolver<RealType>());
    // }
  }
  

Blocks::WavePropagationBlock::WavePropagationBlock(
  int nx, int ny, RealType dx, RealType dy,
  Tools::Float2D<RealType>& h,
  Tools::Float2D<RealType>& hu,
  Tools::Float2D<RealType>& hv
):
  Block(nx, ny, dx, dy, h, hu, hv),
  hNetUpdatesLeft_(nx + 1, ny),
  hNetUpdatesRight_(nx + 1, ny),
  huNetUpdatesLeft_(nx + 1, ny),
  huNetUpdatesRight_(nx + 1, ny),
  hNetUpdatesBelow_(nx, ny + 1),
  hNetUpdatesAbove_(nx, ny + 1),
  hvNetUpdatesBelow_(nx, ny + 1),
  hvNetUpdatesAbove_(nx, ny + 1), 
  wavePropagationSolver_(omp_get_max_threads()),
  wavePropagationSolversSIMD_(omp_get_max_threads())
  {
    // #ifdef WITH_SOLVER_FWAVE
    // for (size_t i = 0; i < omp_get_max_threads(); i++)
    // {
    //   wavePropagationSolver_.push_back(Solvers::FWaveSolver<RealType>());
    //   wavePropatationSolversSIMD_.push_back()
    // }
    // #endif
  }

// void Blocks::WavePropagationBlock::computeNumericalFluxesOneColumn(int tid, int i, int ubj, RealType& maxWaveSpeed)
// {
//   int ub = ubj - ((ubj - 1) % 4);
//   for (int j = 1; j < ub; j += 4) {
//     RealType maxEdgeSpeed = RealType(0.0);
//     // Update the thread-local maximum wave speed
//     wavePropagationSolversSIMD_[tid].computeNetUpdates(
//       &h_[i - 1][j],
//       &h_[i][j],
//       &hu_[i - 1][j],
//       &hu_[i][j],
//       &b_[i - 1][j],
//       &b_[i][j],
//       &hNetUpdatesLeft_[i - 1][j - 1],
//       &hNetUpdatesRight_[i - 1][j - 1],
//       &huNetUpdatesLeft_[i - 1][j - 1],
//       &huNetUpdatesRight_[i - 1][j - 1],
//       maxEdgeSpeed
//     );
//     maxWaveSpeed = std::max(maxWaveSpeed, maxEdgeSpeed);
//   }

//   for (int j = ub; j < ubj; ++j) {
//     RealType maxEdgeSpeed = RealType(0.0);

//     wavePropagationSolver_[tid].computeNetUpdates(
//       h_[i - 1][j],
//       h_[i][j],
//       hu_[i - 1][j],
//       hu_[i][j],
//       b_[i - 1][j],
//       b_[i][j],
//       hNetUpdatesLeft_[i - 1][j - 1],
//       hNetUpdatesRight_[i - 1][j - 1],
//       huNetUpdatesLeft_[i - 1][j - 1],
//       huNetUpdatesRight_[i - 1][j - 1],
//       maxEdgeSpeed
//     );

//     // Update the thread-local maximum wave speed
//     maxWaveSpeed = std::max(maxWaveSpeed, maxEdgeSpeed);
//   }
// }

void Blocks::WavePropagationBlock::computeNumericalFluxes() {
  // Maximum (linearized) wave speed within one iteration
  RealType maxWaveSpeed = RealType(0.0);

  // Compute the net-updates for the vertical edges
  RealType maxWaveSpeedLocal_vec[nx_ + 2];
  RealType maxWaveSpeedLocal_vec2[nx_ + 2];
  maxWaveSpeedLocal_vec2[nx_ + 1] = RealType(0.0);
  RealType maxWaveSpeedLocal1 = RealType(0.0);
  RealType maxWaveSpeedLocal2 = RealType(0.0);

  #pragma omp parallel 
  {

    // #pragma omp single nowait
    // #pragma omp taskloop num_tasks(omp_get_max_threads()*2) nogroup
    auto grainsize_ = (int)((nx_ + 2) / omp_get_num_threads());
    #pragma omp for schedule(guided, grainsize_) 
    for (int i = 1; i < nx_ + 2; i++) {
      maxWaveSpeedLocal_vec[i] = RealType(0.0);
      int ub = (ny_ + 1) - ((ny_ + 1 - 1) % 4);
      auto tid = omp_get_thread_num();
      for (int j = 1; j < ub; j += 4) {
        RealType maxEdgeSpeed = RealType(0.0);

        wavePropagationSolversSIMD_[tid].computeNetUpdates(
          &h_[i - 1][j],
          &h_[i][j],
          &hu_[i - 1][j],
          &hu_[i][j],
          &b_[i - 1][j],
          &b_[i][j],
          &hNetUpdatesLeft_[i - 1][j - 1],
          &hNetUpdatesRight_[i - 1][j - 1],
          &huNetUpdatesLeft_[i - 1][j - 1],
          &huNetUpdatesRight_[i - 1][j - 1],
          maxEdgeSpeed
        );

        // Update the thread-local maximum wave speed
        maxWaveSpeedLocal_vec[i] = std::max(maxWaveSpeedLocal_vec[i], maxEdgeSpeed);

      } 
      for (int j = ub; j < ny_ + 1; ++j) {
        RealType maxEdgeSpeed = RealType(0.0);

        wavePropagationSolver_[tid].computeNetUpdates(
          h_[i - 1][j],
          h_[i][j],
          hu_[i - 1][j],
          hu_[i][j],
          b_[i - 1][j],
          b_[i][j],
          hNetUpdatesLeft_[i - 1][j - 1],
          hNetUpdatesRight_[i - 1][j - 1],
          huNetUpdatesLeft_[i - 1][j - 1],
          huNetUpdatesRight_[i - 1][j - 1],
          maxEdgeSpeed
        );

        // Update the thread-local maximum wave speed
        maxWaveSpeedLocal_vec[i] = std::max(maxWaveSpeedLocal_vec[i], maxEdgeSpeed);

      }
    
    }

    // Compute the net-updates for the horizontal edges
    #pragma omp single nowait
    #pragma omp taskloop num_tasks(omp_get_max_threads()*2) nogroup 
    // #pragma omp for schedule(guided, grainsize_) 
    for (int i = 1; i < nx_ + 1; i++) {
      int ub = (ny_ + 2) - ((ny_ + 2 - 1) % 4);
      maxWaveSpeedLocal_vec2[i] = RealType(0.0);
      auto tid = omp_get_thread_num();

      for (int j = 1; j < ub; j += 4) {
        RealType maxEdgeSpeed = RealType(0.0);
        wavePropagationSolversSIMD_[tid].computeNetUpdates(
          &h_[i][j - 1],
          &h_[i][j],
          &hv_[i][j - 1],
          &hv_[i][j],
          &b_[i][j - 1],
          &b_[i][j],
          &hNetUpdatesBelow_[i - 1][j - 1],
          &hNetUpdatesAbove_[i - 1][j - 1],
          &hvNetUpdatesBelow_[i - 1][j - 1],
          &hvNetUpdatesAbove_[i - 1][j - 1],
          maxEdgeSpeed
        );

        maxWaveSpeedLocal_vec2[i] = std::max(maxWaveSpeedLocal_vec2[i], maxEdgeSpeed);
      }
      for (int j = ub; j < ny_ + 2; ++j) {
        RealType maxEdgeSpeed = RealType(0.0);

        wavePropagationSolver_[tid].computeNetUpdates(
          h_[i][j - 1],
          h_[i][j],
          hv_[i][j - 1],
          hv_[i][j],
          b_[i][j - 1],
          b_[i][j],
          hNetUpdatesLeft_[i - 1][j - 1],
          hNetUpdatesRight_[i - 1][j - 1],
          huNetUpdatesLeft_[i - 1][j - 1],
          huNetUpdatesRight_[i - 1][j - 1],
          maxEdgeSpeed
        );

        maxWaveSpeedLocal_vec2[i] = std::max(maxWaveSpeedLocal_vec2[i], maxEdgeSpeed);

      }
    }
    

  }



  #pragma unroll
  for (int i = 1; i < nx_ + 2; i++) {
    maxWaveSpeed = std::max(maxWaveSpeed, std::max(maxWaveSpeedLocal_vec[i], maxWaveSpeedLocal_vec2[i]));
  }

  }

  if (maxWaveSpeed > 0.00001) {
    // Compute the time step width
    maxTimeStep_ = std::min(dx_ / maxWaveSpeed, dy_ / maxWaveSpeed);

    // Reduce maximum time step size by "safety factor"
    maxTimeStep_ *= RealType(0.4); // CFL-number = 0.5
  } else {
    // Might happen in dry cells
    maxTimeStep_ = std::numeric_limits<RealType>::max();
  }
}

void Blocks::WavePropagationBlock::updateUnknowns(RealType dt) {
  // Update cell averages with the net-updates
  #pragma omp parallel 
  {
    //  auto grainsize_ = (int)((nx_ + 2) / omp_get_num_threads());
    // #pragma omp for schedule(guided, grainsize_) 
    #pragma omp single nowait
    #pragma omp taskloop num_tasks(omp_get_max_threads()*2) nogroup 
    for (int i = 1; i < nx_ + 1; i++) {
      for (int j = 1; j < ny_ + 1; j++) {
        h_[i][j] -= dt / dx_ * (hNetUpdatesRight_[i - 1][j - 1] + hNetUpdatesLeft_[i][j - 1])
                    + dt / dy_ * (hNetUpdatesAbove_[i - 1][j - 1] + hNetUpdatesBelow_[i - 1][j]);
        hu_[i][j] -= dt / dx_ * (huNetUpdatesRight_[i - 1][j - 1] + huNetUpdatesLeft_[i][j - 1]);
        hv_[i][j] -= dt / dy_ * (hvNetUpdatesAbove_[i - 1][j - 1] + hvNetUpdatesBelow_[i - 1][j]);

        if (h_[i][j] < 0) {
          // Zero (small) negative depths
          h_[i][j] = hu_[i][j] = hv_[i][j] = RealType(0.0);
        } else if (h_[i][j] < 0.1) {             // dryTol
          hu_[i][j] = hv_[i][j] = RealType(0.0); // No water, no speed!
        }
      }
    }
  }
}
