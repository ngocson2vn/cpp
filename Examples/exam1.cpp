#include <iostream>
using namespace std;

class shared
{
    static int a;
    int b;

public:
    void set(int i, int j)
    {
        a = i;
        b = j;
    }

    void show();
};

int shared::a; // define a

void shared::show()
{
    cout << "This is static a: " << a;
    cout << "\nThis is non-static b: " << b;
    cout << "\n";
}

int main()
{
    shared x, y;
    x.set(1, 1); // set a to 1
    x.show();
    y.set(2, 2); // change a to 2
    y.show();
    x.show(); /* Here, a has been changed for both x and y 
                 because a is shared by both objects. */
    return 0;
}