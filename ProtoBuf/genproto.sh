SRC_DIR=.
DST_DIR=.
protoc -I=$SRC_DIR --cpp_out=$DST_DIR $SRC_DIR/idl/addressbook.proto