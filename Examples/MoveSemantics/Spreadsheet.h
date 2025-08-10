#include <cstddef>
#include "SpreadsheetCell.h"

class Spreadsheet {
  public:
    Spreadsheet(std::size_t width, std::size_t height);

    // Copy constructor
    Spreadsheet(const Spreadsheet& src);

    // Copy assignment operator
    Spreadsheet& operator=(const Spreadsheet& rhs);

    // Move constructor
    Spreadsheet(Spreadsheet&& src) noexcept;

    // Move assignment operator
    Spreadsheet& operator=(Spreadsheet&& rhs) noexcept;

    void setCellAt(std::size_t x, std::size_t y, const SpreadsheetCell& cell);

    SpreadsheetCell getCellAt(std::size_t x, std::size_t y);

    friend void swap(Spreadsheet& first, Spreadsheet& second) noexcept;

    ~Spreadsheet();

  private:
    std::size_t mWidth = 0;
    std::size_t mHeight = 0;
    SpreadsheetCell** mCells = nullptr;
    
    bool inRange(std::size_t value, std::size_t upper) const;
    void cleanup() noexcept;
    void moveFrom(Spreadsheet& src) noexcept;
};