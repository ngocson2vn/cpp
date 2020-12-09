#include "Spreadsheet.h"

const size_t Spreadsheet::kMaxWidth = 100;
const size_t Spreadsheet::kMaxHeight = 100;

size_t Spreadsheet::sCounter = 0;

Spreadsheet::Spreadsheet(size_t width, size_t height) 
  : mId(sCounter++)
  , mWidth(std::min(width, kMaxWidth))
  , mHeight(std::min(height, kMaxHeight))
{
  std::cout << "Normal constructor" << std::endl;
  mCells = new SpreadsheetCell*[mWidth];
  for (size_t i = 0; i < mWidth; i++) {
    mCells[i] = new SpreadsheetCell[mHeight];
  }
}

Spreadsheet::Spreadsheet(const Spreadsheet& src) : Spreadsheet(src.mWidth, src.mHeight)
{
  for (size_t i = 0; i < mWidth; i++) {
    for (size_t j = 0; j < mHeight; j++) {
      mCells[i][j] = src.mCells[i][j];
    }
  }
}

// Move constructor
Spreadsheet::Spreadsheet(Spreadsheet&& src) noexcept
{
  std::cout << "Move constructor" << std::endl;
  moveFrom(src);
}

void Spreadsheet::setCellAt(size_t x, size_t y, const SpreadsheetCell& cell)
{
  verifyCoordinate(x, y);
  mCells[x][y] = cell;
}

SpreadsheetCell& Spreadsheet::getCellAt(size_t x, size_t y)
{
  verifyCoordinate(x, y);
  return mCells[x][y];
}

void Spreadsheet::verifyCoordinate(size_t x, size_t y) const
{
  if (x >= mWidth || y >= mHeight) {
    throw std::out_of_range("");
  }
}

Spreadsheet& Spreadsheet::operator=(const Spreadsheet& rhs)
{
  if (this == &rhs) {
    return *this;
  }

  // Free the old memory
  for (size_t i = 0; i < mWidth; i++) {
    delete[] mCells[i];
  }
  delete[] mCells;
  mCells = nullptr;

  // Allocate new memory
  mWidth = rhs.mWidth;
  mHeight = rhs.mHeight;

  mCells = new SpreadsheetCell*[mWidth];
  for (size_t i = 0; i < mWidth; i++) {
    mCells[i] = new SpreadsheetCell[mHeight];
  }

  for (size_t i = 0; i < mWidth; i++) {
    for (size_t j = 0; j < mHeight; j++) {
      mCells[i][j] = rhs.mCells[i][j];
    }
  }

  return *this;
}

void Spreadsheet::cleanup() noexcept
{
  for (size_t i = 0; i < mWidth; i++) {
    delete[] mCells[i];
  }
  delete[] mCells;
  mCells = nullptr;

  mWidth = 0;
  mHeight = 0;
}

void Spreadsheet::moveFrom(Spreadsheet& src) noexcept
{
  // Shallow copy of data
  mWidth = src.mWidth;
  mHeight = src.mHeight;
  mCells = src.mCells;

  // Reset the source object, because ownership has been moved
  src.mWidth = 0;
  src.mHeight = 0;
  src.mCells = nullptr;
}

// Move assignment operator
Spreadsheet& Spreadsheet::operator=(Spreadsheet&& rhs) noexcept
{
  // Check for self-assignment
  if (this == &rhs) {
    return *this;
  }

  // Free this object's memory
  cleanup();

  // Transfer memory ownership
  moveFrom(rhs);

  return *this;
}

Spreadsheet::~Spreadsheet()
{
  for (size_t i = 0; i < mWidth; i++) {
    delete[] mCells[i];
  }
  delete[] mCells;
  mCells = nullptr;
  mWidth = 0;
  mHeight = 0;
}