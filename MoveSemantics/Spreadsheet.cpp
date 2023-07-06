#include "Spreadsheet.h"
#include <utility>
#include <iostream>

void swap(Spreadsheet& first, Spreadsheet& second) noexcept {
  std::swap(first.mWidth, second.mWidth);
  std::swap(first.mHeight, second.mHeight);
  std::swap(first.mCells, second.mCells);
}

Spreadsheet::Spreadsheet(std::size_t width, std::size_t height) {
  std::cout << "++ Constructor Spreadsheet::Spreadsheet(std::size_t width, std::size_t height)" << std::endl;
  mWidth = width;
  mHeight = height;
  mCells = new SpreadsheetCell*[mWidth];
  for (int i = 0; i < mWidth; i++) {
    mCells[i] = new SpreadsheetCell[mHeight];
  }
}

// Copy constructor
Spreadsheet::Spreadsheet(const Spreadsheet& src) : Spreadsheet(src.mWidth, src.mHeight) {
  std::cout << "++ Copy constructor Spreadsheet::Spreadsheet(const Spreadsheet& src)" << std::endl;
  for (int i = 0; i < mWidth; i++) {
    for (int j = 0; j < mHeight; j++) {
      mCells[i][j] = src.mCells[i][j];
    }
  }
}

// Copy assignment operator
Spreadsheet& Spreadsheet::operator=(const Spreadsheet& rhs) {
  std::cout << "++ Copy assignment operator Spreadsheet& Spreadsheet::operator=(const Spreadsheet& rhs)" << std::endl;
  if (this == &rhs) {
    return *this;
  }

  // Free dynamically allocated memory that this object is the owner.
  cleanup();

  // Deep copy
  mWidth = rhs.mWidth;
  mHeight = rhs.mHeight;
  mCells = new SpreadsheetCell*[mWidth];
  for (int i = 0; i < mWidth; i++) {
    mCells[i] = new SpreadsheetCell[mHeight];
    for (int j = 0; j < mHeight; j++) {
      mCells[i][j] = rhs.mCells[i][j];
    }
  }

  return *this;
}

// Move constructor
Spreadsheet::Spreadsheet(Spreadsheet&& src) noexcept {
  std::cout << ">> Move constructor Spreadsheet::Spreadsheet(Spreadsheet&& src)" << std::endl;
  swap(*this, src);
}

// Move assignment operator
Spreadsheet& Spreadsheet::operator=(Spreadsheet&& rhs) noexcept {
  std::cout << ">> Move assignment operator Spreadsheet& Spreadsheet::operator=(Spreadsheet&& rhs)" << std::endl;
  if (this == &rhs) {
    return *this;
  }

  // Move constructor
  Spreadsheet temp(std::move(rhs));

  // Swap
  swap(*this, temp);

  return *this;
}

void Spreadsheet::setCellAt(std::size_t x, std::size_t y, const SpreadsheetCell& cell) {
    
}

SpreadsheetCell Spreadsheet::getCellAt(std::size_t x, std::size_t y) {
  return mCells[x][y];
}

bool Spreadsheet::inRange(std::size_t value, std::size_t upper) const {
  return true;
}

void Spreadsheet::cleanup() noexcept {
    for (std::size_t i = 0; i < mWidth; i++) {
      delete[] mCells[i];
    }

    delete mCells;
    mCells = nullptr;
    mWidth = 0;
    mHeight = 0;
}

void Spreadsheet::moveFrom(Spreadsheet& src) noexcept {
  // Shallow copy of data
  mWidth = src.mWidth;
  mHeight = src.mHeight;
  mCells = src.mCells;

  // Reset the source object, because the ownership has been moved!
  src.mWidth = 0;
  src.mHeight = 0;
  src.mCells = nullptr;
}

Spreadsheet::~Spreadsheet() {
  cleanup();
}
