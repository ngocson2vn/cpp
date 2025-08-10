#include <iostream>
#include <vector>
#include "Spreadsheet.h"

Spreadsheet createSpreadsheet() {
  return Spreadsheet(10, 10);
}

SpreadsheetCell createSpreadsheetCell(const std::string& name, double value) {
  return SpreadsheetCell(name, value);
}

template <typename T>
class TypeDetector;

// Caller can pass either a lvalue or a rvalue
// cell is a const lvalue reference
void checkSpreadsheetCellLvalue(const SpreadsheetCell& cell) {
  // TypeDetector<decltype(cell)> td;
  std::cout << "Cell address: " << &cell << std::endl;
  std::cout << "Cell value: " << cell.getString() << std::endl;
}

void validateSpreadsheetCellRvalue(SpreadsheetCell&& cell) {
  std::cout << "Cell address: " << &cell << std::endl;
  std::cout << "Cell value: " << cell.getString() << std::endl;
}

// Caller can pass only rvalue
// cell is a rvalue reference
void checkSpreadsheetCellRvalue(SpreadsheetCell&& cell) {
  std::cout << "Cell address: " << &cell << std::endl;
  std::cout << "Cell value: " << cell.getString() << std::endl;
  // TypeDetector<decltype(cell)> td;
  // validateSpreadsheetCellRvalue(cell);
}

int main() {
  // std::vector<Spreadsheet> vec;

  // for (int i = 0; i < 2; i++) {
  //   std::cout << "Iteration " << i << std::endl;
  //   vec.push_back(Spreadsheet(100, 100));
  //   std::cout << std::endl;
  // }

  // SpreadsheetCell cell1(10);
  // SpreadsheetCell cell2(20);
  // SpreadsheetCell cell3 = cell1 + 5;
  // SpreadsheetCell testCell1 = createSpreadsheetCell();
  
  // SpreadsheetCell testCell2;
  // testCell2 = createSpreadsheetCell();


  // SpreadsheetCell lvalueCell = createSpreadsheetCell("Cell_1", 100.0);
  // std::cout << "Cell address = " << &lvalueCell << std::endl;
  // checkSpreadsheetCellLvalue(lvalueCell);
  // std::cout << std::endl;

  // checkSpreadsheetCellRvalue(createSpreadsheetCell("Cell_1", 100.0));
  checkSpreadsheetCellLvalue(createSpreadsheetCell("Cell_1", 100.0));

  std::cout << "== END main ==" << std::endl;

  return 0;
}