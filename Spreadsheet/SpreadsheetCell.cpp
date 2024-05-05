#include "SpreadsheetCell.h"
#include <cstdlib>

SpreadsheetCell::SpreadsheetCell(const std::string& name, double initialValue) {
  mId = std::rand();
  std::cout << "SpreadsheetCell Normal Constructor; this=" << this << " mId=" << mId << std::endl;
  setName(name);
  setValue(initialValue);
}

SpreadsheetCell::SpreadsheetCell(const SpreadsheetCell& src) noexcept : mValue(src.mValue) {
  std::cout << "SpreadsheetCell Copy Constructor" << std::endl;
}

SpreadsheetCell::SpreadsheetCell(SpreadsheetCell&& rhs) noexcept {
  std::cout << "START SpreadsheetCell Move Constructor; from mId=" << rhs.mId << std::endl;
  mId = std::rand();
  mName = std::move(rhs.mName);
  mValue = rhs.mValue;
  rhs.mValue = 0;
  std::cout << "END SpreadsheetCell Move Constructor; to mId=" << mId << std::endl;
}

SpreadsheetCell::~SpreadsheetCell() noexcept {
  std::cout << "SpreadsheetCell Destructor, mId=" << mId << std::endl;
  mValue = 0;
}

void SpreadsheetCell::setName(const std::string& name) {
  mName = name;
}

void SpreadsheetCell::setValue(double inValue) {
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
  return SpreadsheetCell(mName, getValue() + rhs.getValue());
}

SpreadsheetCell& SpreadsheetCell::operator=(const SpreadsheetCell& rhs) noexcept {
  if (this == &rhs) {
    return *this;
  }

  mValue = rhs.mValue;
  return *this;
}

SpreadsheetCell& SpreadsheetCell::operator=(SpreadsheetCell&& rhs) noexcept {
  std::cout << "SpreadsheetCell Move Assignment Operator" << std::endl;
  if (this == &rhs) {
    return *this;
  }

  mValue = rhs.mValue;
  rhs.mValue = 0;
  return *this;
}