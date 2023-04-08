#include <cstring>
#include <algorithm>
#include <stdio.h>

class string {
private:
	char* data;

public:
	explicit string(const char* p) {
		log("+ constructor");
		size_t size = strlen(p) + 1;
		data = new char[size];
		memcpy(data, p, size);
	}

	~string() {
		log("- destructor");
		if (data) {
			log("\t-> delete allocated heap data");
			delete[] data;
			data = NULL;
		}
	}

	string(const string& that) {
		log("copy constructor");
		size_t size = strlen(that.data) + 1;
		data = new char[size];
		memcpy(data, that.data, size);
		*(data + size - 1) = '\0';
	}

	string(string&& that) {
		log("MOVE CONSTRUCTOR");
		data = that.data;
		that.data = NULL;
	}

	string& operator=(string that) {
		log("copy assignment operator");
		std::swap(data, that.data);
		return *this;
	}

	string operator+(string that) {
		log("concatenate operator");
		size_t len = strlen(data);
		size_t len2 = strlen(that.data);
		size_t size = len + len2 + 1;

		char* retdata = new char[size];

		memcpy(retdata, data, len);
		memcpy(retdata + len, that.data, len2);
		*(retdata + size - 1) = '\0';

		string ret(retdata);

		return ret;
	}

	void print() const {
		printf("\n[%08X] data: %s\n\n", this, data);
	}

	void log(const char* msg) {
		printf("[%08X] %s\n", this, msg);
	}
};

int TestRvalueMoveConstructor() {
//int main() {
	string a("a");
	string b("b");

	string x("Hello");
	string y("World!");

	a = b;
	a.print();

	printf("\n");
	x + y;
	//a = x + y;
	a.print();

	return getchar();
}