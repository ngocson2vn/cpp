#include <iostream>
#include "common.h"

class DataClass {
  public:
    DataClass() = default;

    DataClass(int initialValue) {
      std::cout << "DataClass Normal Constructor: " << this << std::endl;
      mValue = initialValue;
    }

    // Copy
    DataClass(const DataClass& m) {
      std::cout << "DataClass Copy Constructor\n";
      mValue = m.mValue;
    }

    DataClass& operator=(const DataClass& m) {
      std::cout << "DataClass Copy Assignment Operator\n";
      if (this == &m) {
        return *this;
      }

      mValue = 0;
      mValue = m.mValue;
    }

    // Move
    DataClass(DataClass&& m) {
      std::cout << "DataClass Move Constructor\n";
      mValue = m.mValue;
      m.mValue = 0;
    }

    DataClass& operator=(DataClass&& m) {
      std::cout << "DataClass Move Assignment Operator\n";
      if (this == &m) {
        return *this;
      }

      mValue = 0;
      mValue = m.mValue;
      m.mValue = 0;
      return *this;
    }

    void printValue() const {
      std::cout << mValue << std::endl;
    }

    ~DataClass() {
      std::cout << "DataClass Destructor: " << this << std::endl;
      mValue = 0;
    }

  private:
    int mValue;
};

class Widget {
  public:
    Widget() {
      std::cout << "Widget Normal Constructor: " << this << std::endl;
    }

    Widget(const std::string& name) : name_(name) {
      std::cout << "Widget Normal Constructor: " << this << std::endl;
    }

    // template <typename T>
    // void setData(T&& newData) {
    //   data_ = std::move(newData);
    // }
    
    void setData(const DataClass& newData) {
      data_ = newData;
    }

    void setData(DataClass&& newData) {
      data_ = std::move(newData);
    }

    ~Widget() {
      std::cout << "Widget Destructor: " << this << std::endl;
      name_.clear();
    }

    const std::string& getName() const {
      std::cout << "&name_: " << &name_ << std::endl;
      return name_;
    }
  
  private:
    std::string name_;
    DataClass data_;
};

DataClass getWidgetData() {
  return DataClass(10);
}

Widget getWidget() {
  return Widget("TEST");
}

int main() {
  // Widget w;
  // auto d = getWidgetData();
  // w.setData();
  auto& widgetName = getWidget().getName();
  std::cout << "widgetName's type: " << type_name<decltype(widgetName)>() << std::endl;
  std::cout << "&widgetName: " << &widgetName << std::endl;
  std::cout << "widgetName's value: " << widgetName << std::endl;
}
