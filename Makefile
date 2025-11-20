CXX = g++
CXXFLAGS = -std=c++20 -O3 -Wall -Wextra

TARGET = graph_gen
SRCS = main.cpp generate.cpp

all: $(TARGET)

$(TARGET):
	$(CXX) $(CXXFLAGS) $(SRCS) -o $(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: all clean
