#include <iostream>
#include <vector>
#include "Spreadsheet.h"

Spreadsheet createSpreadsheet()
{
  return Spreadsheet(10, 10);
}

SpreadsheetCell createSpreadsheetCell() {
  return SpreadsheetCell(100);
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
  SpreadsheetCell testCell1 = createSpreadsheetCell();
  
  // SpreadsheetCell testCell2;
  // testCell2 = createSpreadsheetCell();

  return 0;
}