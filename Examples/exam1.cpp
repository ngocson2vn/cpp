#include <iostream>
#include <limits>
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

    cout << "max num_elements: " << (std::numeric_limits<size_t>::max() / sizeof(float)) << std::endl;

    std::string head_name = "fc_ecom_cart_action_sim_logspan:0";
    size_t ret = head_name.find("fc_ecom_cart_action_sim_logspan", 0);
    std::cout << "head name: " << head_name << std::endl;
    std::cout << "ret: " << ret << std::endl;

    return 0;
}