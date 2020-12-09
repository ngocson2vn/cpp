#include <string>

class SpreadsheetCell
{
public:
  SpreadsheetCell() = default;
  SpreadsheetCell(double initialValue);
  void setValue(double inValue);
  double getValue() const;
  void setString(const std::string& inString);
  std::string getString() const;
  SpreadsheetCell operator+(const SpreadsheetCell& cell) const;

private:
  std::string doubleToString(double inValue) const;
  double stringToDouble(const std::string& inString) const;
  double mValue;
};