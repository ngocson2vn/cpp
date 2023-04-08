#include <iostream>
#include "Intvec.h"

Intvect::Intvect(size_t num) : m_size(num), m_data(new int[m_size]) {
	log("constructor");
}

Intvect::Intvect(const Intvect& other) : m_size(other.m_size), m_data(new int[m_size]) {
	log("copy constructor");
	for (size_t i = 0; i < m_size; i++) {
		m_data[i] = other.m_data[i];
	}
}

Intvect& Intvect::operator=(const Intvect& other) {
	log("copy assignment operator");
	Intvect temp(other);
	std::swap(m_size, temp.m_size);
	std::swap(m_data, temp.m_data);

	return *this;
}

Intvect& Intvect::operator = (Intvect&& other) {
	log("move assignment operator");
	std::swap(m_size, other.m_size);
	std::swap(m_data, other.m_data);

	return *this;
}

void Intvect::log(const char* msg) {
	std::cout << "[" << this << "] " << msg << std::endl;
}

Intvect::~Intvect() {
	log("destructor");
	if (m_data) {
		delete[] m_data;
		m_data = nullptr;
	}
}