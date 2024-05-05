#!/bin/bash

g++ -c main.cpp student_factory.cpp dummy.cpp
g++ -o main main.o student_factory.o dummy.o

g++ -c main.cpp student_factory.cpp university_student.cpp

# g++ -o main main.o student_factory.o dummy.o university_student.o
# g++ -o main main.o student_factory.o university_student.o