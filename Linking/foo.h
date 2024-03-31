class Foo {
  public:
    Foo();
  
    void setFooName(const char* name);

    const char* getFooName();

  private:
    char* name_;
};

extern Foo foo;
