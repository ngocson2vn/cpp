#include <iostream>
#include <limits>
<<<<<<< HEAD
=======
using namespace std;
>>>>>>> 32f7313692337423c2ca65b1053c348790211df4

size_t get_bs() {
    int64_t bs = -1;
    std::cout << "bs1 = " << bs << std::endl;
    return bs;
}

int main()
{
<<<<<<< HEAD
    size_t maxvalue = std::numeric_limits<size_t>::max();
    std::cout << "maxvalue = " << maxvalue << std::endl;
    size_t bs = get_bs();
    std::cout << "bs2 = " << bs << std::endl;
=======
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

>>>>>>> 32f7313692337423c2ca65b1053c348790211df4
    return 0;
}