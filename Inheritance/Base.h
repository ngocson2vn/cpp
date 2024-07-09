class Base {
  public:
    virtual void someMethod();
    virtual void setIntValue(int value);
    virtual void showIntValue();
  protected:
    int mProtectedInt;
  private:
    void increaseIntValue();
    int mPrivateInt;
};