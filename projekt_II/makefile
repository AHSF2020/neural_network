# Application target.
TARGET:= project2

# Source files used in the application.
SOURCE_FILES := src/main.cpp \
                src/dense_layer.cpp \
				src/button.cpp \
                src/gpiod_utils.c \
			     src/led.cpp \

# Include directories.
INCLUDE_DIRS := include

# Additional compiler flags.
COMPILER_FLAGS := -Wall -Werror

# Linked libraries.
LINKED_LIBS := gpiod

# Builds and runs the application as default.
default: build run

# Builds the application.
build:
	@g++ $(SOURCE_FILES) -o $(TARGET) -I $(INCLUDE_DIRS) $(COMPILER_FLAGS) -l $(LINKED_LIBS)

# Runs the application.
run:
	@./$(TARGET)

# Cleans the application.
clean:
	@rm -f $(TARGET)
