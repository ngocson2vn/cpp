#pragma once

template <typename T>
HVec<T> torchGemm(const HVec<T>& A,
                  const HVec<T>& B,
                  const uint32_t M,
                  const uint32_t N,
                  const uint32_t K);

template <typename T>
HVec<T> torchSoftmax(const HVec<T>& hInp,
                     const uint32_t M,
                     const uint32_t N);
