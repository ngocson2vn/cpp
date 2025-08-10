main: Spreadsheet.o SpreadsheetCell.o main.cpp
	/usr/bin/g++ -g -std=c++14 *.o main.cpp -o main

# Spreadsheet.o: Spreadsheet.h Spreadsheet.cpp
# 	/usr/bin/g++ -g -std=c++14 -fno-elide-constructors -c Spreadsheet.cpp

# SpreadsheetCell.o: SpreadsheetCell.h SpreadsheetCell.cpp
# 	/usr/bin/g++ -g -std=c++14 -fno-elide-constructors -c SpreadsheetCell.cpp

Spreadsheet.o: Spreadsheet.h Spreadsheet.cpp
	/usr/bin/g++ -g -std=c++14 -c Spreadsheet.cpp

SpreadsheetCell.o: SpreadsheetCell.h SpreadsheetCell.cpp
	/usr/bin/g++ -g -std=c++14 -c SpreadsheetCell.cpp