#include <iostream>

class MyClass
{
    public:
        MyClass() = default;

        MyClass(int initialValue)
        {
            mValue = initialValue;
        }

        void printValue() const
        {
            std::cout << mValue << std::endl;
        }

    private:
        int mValue;
};

int main()
{
    MyClass m1;
    m1.printValue();

    MyClass m2(10);
    m2.printValue();
}