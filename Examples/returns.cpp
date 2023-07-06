#include <iostream>
#include <format>

class Widget {
  public:
    Widget() = default;

    Widget(int initialValue) {
      std::cout << "Widget Normal Constructor: " << this << std::endl;
      mValue = initialValue;
    }

    // Copy
    Widget(const Widget& m) {
      std::cout << "Widget Copy Constructor: " << this << std::endl;
      mValue = m.mValue;
    }

    Widget& operator=(const Widget& m) {
      std::cout << "Widget Copy Assignment Operator\n";
      if (this == &m) {
        return *this;
      }

      mValue = 0;
      mValue = m.mValue;
    }

    // Move
    Widget(Widget&& m) {
      std::cout << "Widget Move Constructor\n";
      mValue = m.mValue;
      m.mValue = 0;
    }

    Widget& operator=(Widget&& m) {
      std::cout << "Widget Move Assignment Operator\n";
      if (this == &m) {
        return *this;
      }

      mValue = 0;
      mValue = m.mValue;
      m.mValue = 0;
      return *this;
    }

    void printValue() const {
      std::cout << "[" << this << "] mValue = " << mValue << std::endl;
    }

    ~Widget() {
      std::cout << "Widget Destructor: " << this << std::endl;
      mValue = 0;
    }

    void doubleValue() {
      mValue *= 2;
    }

  private:
    int mValue;
};

Widget& buildWidget() {
  std::cout << "Build a Widget\n";
  Widget tmpWidget(100);
  return tmpWidget;
}

Widget w(10);

Widget& changeWidget() {
  w.doubleValue();
  return w;
}

template <typename Container, typename Index>
decltype(auto) get_element(Container&& c, Index i) {
  return c[i];
}

std::vector<int> getScores() {
  std::vector<int> tmpScores = {1, 2, 3};
  std::cout << "&tmpScores[1]: " << &tmpScores[1] << std::endl;
  return tmpScores;
}

int main() {
  // Widget w = buildWidget();
  // w.printValue();

  w.printValue();
  std::cout << std::endl;
  
  Widget w2 = changeWidget();
  w2.printValue();
  std::cout << std::endl;

  Widget& w3 = changeWidget();
  w3.printValue();
  std::cout << std::endl;

  auto e = get_element(getScores(), 1);
  std::cout << "&e: " << &e << std::endl;
  std::cout << "e: " << e << std::endl;
}
