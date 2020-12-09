#include <cstddef>
#include <iostream>
#include <algorithm>
#include "SpreadsheetCell.h"

class SpreadsheetApplication; // forward declaration

class Spreadsheet
{
public:
  static const size_t kMaxWidth;
  static const size_t kMaxHeight;

  Spreadsheet(size_t width, size_t height);
  Spreadsheet(const Spreadsheet& src);
  Spreadsheet(Spreadsheet&& src) noexcept;
  ~Spreadsheet();
  void setCellAt(size_t x, size_t y, const SpreadsheetCell& cell);
  SpreadsheetCell& getCellAt(size_t x, size_t y);
  size_t getId() const;
  Spreadsheet& operator=(const Spreadsheet& rhs);
  Spreadsheet& operator=(Spreadsheet&& rhs) noexcept;

private:
  void cleanup() noexcept;
  void moveFrom(Spreadsheet& src) noexcept;
  void verifyCoordinate(size_t x, size_t y) const;
  size_t mWidth = 0;
  size_t mHeight = 0;
  SpreadsheetCell** mCells = nullptr;
  size_t mId = 0;
  // SpreadsheetApplication& mTheApp;

  static size_t sCounter;
};