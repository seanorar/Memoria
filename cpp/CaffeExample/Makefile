#Makefile for Impresee
#

CXX := g++
CFLAGS :=  -std=c++11  -O0 -ggdb  -Wall -Wextra  -Wno-unknown-pragmas -fpic  -Wno-unused-local-typedefs
PACKAGES :=  cvision jmsr flann opencv caffe
INCPATH += $(shell pkg-config --cflags $(PACKAGES))
LIBS := $(shell pkg-config --libs  $(PACKAGES)) -lstdc++
BUILD_DIR = build
LD_FLAGS := $(LIBS)
CFLAGS += $(INCPATH)

HEADER_FILES := $(shell find $(INCLUDE_DIR) -type f -and \( -name \*.h -or -name \*.hpp \) )
TEST_FILES	:= $(shell find . -type f -name \*.cpp)
TEST_FILES	:= $(basename $(TEST_FILES))

TARGET_TEST	:= $(addprefix $(BUILD_DIR)/, $(TEST_FILES))

.PHONY: all

all: $(TARGET_TEST)


$(BUILD_DIR)/%.o: %.cpp
	mkdir -p "$(@D)"
	$(CXX) -c  $(CFLAGS)  -o $@ $<  
        
$(BUILD_DIR)/%: $(BUILD_DIR)/%.o
	$(CXX)  $(CFLAGS) -o $@ $< $(LD_FLAGS)

clean:
	rm -rf "$(BUILD_DIR)"



