#include <iostream>
#include <string>

class SpreadsheetCell
{
public:
  SpreadsheetCell() = default;
  SpreadsheetCell(double initialValue);
  SpreadsheetCell(const SpreadsheetCell& rhs) noexcept;
  SpreadsheetCell(SpreadsheetCell&& rhs) noexcept;
  void setValue(double inValue);
  double getValue() const;
  void setString(const std::string& inString);
  std::string getString() const;
  SpreadsheetCell operator+(const SpreadsheetCell& rhs) const;
  SpreadsheetCell& operator=(const SpreadsheetCell& rhs) noexcept;
  SpreadsheetCell& operator=(SpreadsheetCell&& rhs) noexcept;

private:
  std::string doubleToString(double inValue) const;
  double stringToDouble(const std::string& inString) const;
  double mValue;
};