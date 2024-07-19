cpp \
-O0 \
-std=c++17 \
-D__SSE4_2__=1 \
-Ifolly/include \
-Iopenssl/include \
-Iboost/include \
main.cc > main.cc.cpp