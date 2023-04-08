#include "SpreadsheetCell.h"

SpreadsheetCell::SpreadsheetCell(double initialValue)
{
  setValue(initialValue);
}

SpreadsheetCell::SpreadsheetCell(const SpreadsheetCell& src) noexcept : mValue(src.mValue) {
  std::cout << "Executing SpreadsheetCell Copy Constructor" << std::endl;
}

SpreadsheetCell::SpreadsheetCell(SpreadsheetCell&& rhs) noexcept {
  std::cout << "Executing SpreadsheetCell Move Constructor" << std::endl;
  mValue = rhs.mValue;
  rhs.mValue = 0;
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

SpreadsheetCell SpreadsheetCell::operator+(const SpreadsheetCell& rhs) const
{
  return SpreadsheetCell(getValue() + rhs.getValue());
}

SpreadsheetCell& SpreadsheetCell::operator=(const SpreadsheetCell& rhs) noexcept {
  if (this == &rhs) {
    return *this;
  }

  mValue = rhs.mValue;
  return *this;
}

SpreadsheetCell& SpreadsheetCell::operator=(SpreadsheetCell&& rhs) noexcept {
  std::cout << "Executing SpreadsheetCell Move Assignment Operator" << std::endl;
  if (this == &rhs) {
    return *this;
  }

  mValue = rhs.mValue;
  rhs.mValue = 0;
  return *this;
}