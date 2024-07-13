/usr/bin/g++ \
-g \
-std=c++17 \
-Ifolly/include \
-Iopenssl/include \
-Iboost/include \
-Lopenssl/lib \
-Lboost/lib \
-Lfolly/lib \
-o main \
main.cc \
-lfolly