#ifndef INTVECT_H
#define INTVECT_H

class Intvect {
public:
	explicit Intvect(size_t num = 0);
	Intvect(const Intvect& other);
	Intvect& operator=(const Intvect& other);
	Intvect& operator=(Intvect&& other);
	~Intvect();

private:
	size_t m_size;
	int* m_data;
	void log(const char* msg);
};

#endif