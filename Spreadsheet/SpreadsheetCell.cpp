#include "SpreadsheetCell.h"

SpreadsheetCell::SpreadsheetCell(double initialValue)
{
  setValue(initialValue);
}

void SpreadsheetCell::setValue(double inValue)
{
  mValue = inValue;
}

double SpreadsheetCell::getValue() const
{
  return mValue;
}

void SpreadsheetCell::setString(const std::string& inString)
{
  mValue = stringToDouble(inString);
}

std::string SpreadsheetCell::getString() const
{
  return doubleToString(mValue);
}

std::string SpreadsheetCell::doubleToString(double inValue) const
{
  return std::to_string(inValue);
}

double SpreadsheetCell::stringToDouble(const std::string& inString) const
{
  return std::strtod(inString.data(), nullptr);
}

SpreadsheetCell SpreadsheetCell::operator+(const SpreadsheetCell& cell) const
{
  return SpreadsheetCell(getValue() + cell.getValue());
}