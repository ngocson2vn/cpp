# template keyword
```C++
template <typename T>
struct foo {
  template <typename U>
  void bar() { }
};

template <typename T>
void func(foo<T> f) {
  f.bar<float>();
}

int main(int argv, char** argc) {
  return 0;
}
```

Compile it:
```Bash
$ make k1
clang++ -o k1 template_keyword1.cpp
template_keyword1.cpp:9:5: error: use 'template' keyword to treat 'bar' as a dependent template name
  f.bar<float>();
    ^
    template 
1 error generated.
make: *** [Makefile:23: k1] Error 1
```

## Two-phase name lookup
C++ standard:
>$14.2/4: When the name of a member template specialization appears after . or -> in a postfix-expression, or after nested-name-specifier in a qualified-id, and the postfix-expression or qualified-id explicitly depends on a template-parameter (14.6.2), the member template name must be prefixed by the keyword template. Otherwise the name is assumed to name a non-template.

Being faced with this error message it may not be clear what the problem is. The reason for the error can actually be found in **two-phase name lookup**; the rule that every template is compiled in two phases, firstly for general syntax and again once any dependent names (names that depend on a template parameter) are known.  

**Phase 1**
Name Lookup: This is done during the template definition. The compiler looks up names in the context of the template but does not yet fully instantiate the template or resolve dependent names. The compiler only checks the syntactic structure and sees that `f.bar<float>()` is a dependent name (due to `f` being a template parameter type `foo<T>`) but does not fully resolve what `bar` actually is.

**Phase 2**
Name Resolution and Template Instantiation: This occurs when the template is actually instantiated or when the compiler needs to fully resolve dependent names. The compiler resolves the dependent names using the complete context, including checking whether `bar` is a member function template and handling the specifics of template arguments. But the compiler cannot deduce `bar` because it does not yet know what type `f` is and thus requires explicit indication using the `template` keyword.

In order to instruct the compiler that this is, in fact, a call to a member function template specialisation, it is necessary to add the `template` keyword immediately after the `.` operator:  
```C++
template <typename T>
struct foo {
  template <typename U>
  void bar() { }
};

template <typename T>
void func(foo<T> f) {
  f.template bar<float>();
}
```

# typename keyword
```C++
template <typename T>
struct type_or_value;

template <>
struct type_or_value<int> {
  static const bool tv = true;
};

template <>
struct type_or_value<float> {
  using tv = float;
};

template <typename T>
void func() {
  using t = type_or_value<T>::tv;
  bool v = type_or_value<T>::tv;
}

int main(int argc, char** argv) {
  return 0;
}
```

Compile it:
```Bash
$ make k3
clang++ -o k3 typename_keyword1.cpp
typename_keyword1.cpp:16:13: error: missing 'typename' prior to dependent type name 'type_or_value<T>::tv'
  using t = type_or_value<T>::tv;
            ^~~~~~~~~~~~~~~~~~~~
            typename 
1 error generated.
make: *** [Makefile:32: k3] Error 1
```