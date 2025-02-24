CXX       = mpicxx
CXXFLAGS  = -O3
SRCS      = $(wildcard *.cpp)
TARGETS   = $(SRCS:.cpp=)

all: $(TARGETS)

%: %.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

clean:
	rm -f $(TARGETS)

.PHONY: all clean