#include <iostream>
#include <string>

class SpreadsheetCell {
public:
  SpreadsheetCell() = default;
  SpreadsheetCell(const std::string& name, double initialValue);
  SpreadsheetCell(const SpreadsheetCell& rhs) noexcept;
  SpreadsheetCell(SpreadsheetCell&& rhs) noexcept;
  ~SpreadsheetCell() noexcept;
  void setName(const std::string& name);
  void setValue(double inValue);
  double getValue() const;
  void setString(const std::string& inString);
  std::string getString() const;
  SpreadsheetCell operator+(const SpreadsheetCell& rhs) const;
  SpreadsheetCell& operator=(const SpreadsheetCell& rhs) noexcept;
  SpreadsheetCell& operator=(SpreadsheetCell&& rhs) noexcept;

private:
  int mId;
  std::string mName;
  std::string doubleToString(double inValue) const;
  double stringToDouble(const std::string& inString) const;
  double mValue;
};