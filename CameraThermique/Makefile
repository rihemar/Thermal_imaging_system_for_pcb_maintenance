I2C_MODE = LINUX
I2C_LIBS = 
SRC_DIR = examples/src/
BUILD_DIR = examples/
LIB_DIR = $(SRC_DIR)lib/


examples = test testRihem sdlscale claude GUI oled testoled 
examples_objects = $(addsuffix .o,$(addprefix $(SRC_DIR), $(examples)))
examples_output = $(addprefix $(BUILD_DIR), $(examples))

#debugging enabled by default
CXXFLAGS+=-DDEBUG -g #-DNOTF
# SDL2 auto-detection
SDL2_CFLAGS  := $(shell pkg-config --cflags sdl2 )
SDL2_LDFLAGS := $(shell pkg-config --libs sdl2 )

SDL2_TTF_CFLAGS  := $(shell pkg-config --cflags SDL2_ttf )
SDL2_TTF_LIBS    := $(shell pkg-config --libs SDL2_ttf)

CXXFLAGS += $(SDL2_TTF_CFLAGS)
LDLIBS   += $(SDL2_TTF_LIBS)

CXXFLAGS += $(SDL2_CFLAGS)
LDLIBS   += $(SDL2_LDFLAGS)
LDLIBS += $(shell sdl2-config --libs)
LDLIBS += -lSSD1306_OLED_RPI -lbcm2835 -lrt
CXXFLAGS += -Iheaders
CXXFLAGS += -lSSD1306_OLED_RPI -lbcm2835 -lrt

PREFIX = /usr/local

all: libMLX90640_API.a libMLX90640_API.so examples


examples: $(examples_output)

libMLX90640_API.so: functions/MLX90640_API.o functions/MLX90640_$(I2C_MODE)_I2C_Driver.o $(UG82_OBJ)
	$(CXX) -fPIC -shared $^ -o $@ $(I2C_LIBS)

libMLX90640_API.a: functions/MLX90640_API.o functions/MLX90640_$(I2C_MODE)_I2C_Driver.o $(UG82_OBJ)
	ar rcs $@ $^
	ranlib $@

functions/MLX90640_API.o functions/MLX90640_RPI_I2C_Driver.o functions/MLX90640_LINUX_I2C_Driver.o : CXXFLAGS+=-fPIC -I headers -shared $(I2C_LIBS)

# -Wno-deprecated-declarations fixes deprecation warnings affecting LibAV v4.0+ in video.cpp
# -Wno-format-extra-args ignores the ANSI colour formatting in test.cpp
$(examples_objects) : CXXFLAGS+=-std=c++11 -Wall  -Wno-format-extra-args -Wno-deprecated-declarations #-Werror

$(examples_output) : CXXFLAGS+=-I. -std=c++11 


examples/src/lib/interpolate.o : CC=$(CXX) -std=c++11

examples/src/sdlscale.o : CXXFLAGS+=$(shell sdl2-config --cflags )
examples/src/GUI.o : CXXFLAGS+=$(shell sdl2-config --cflags )


$(BUILD_DIR)sdlscale: $(SRC_DIR)sdlscale.o libMLX90640_API.a
	$(CXX) -L/home/pi/mlx90640-library $^ -o $@ $(I2C_LIBS) $(LDLIBS)
$(BUILD_DIR)hotspot: $(SRC_DIR)hotspot.o $(LIB_DIR)fb.o libMLX90640_API.a
	$(CXX) -L/home/pi/mlx90640-library $^ -o $@ $(I2C_LIBS)

$(BUILD_DIR)test: $(SRC_DIR)test.o libMLX90640_API.a  /usr/local/lib/libSSD1306_OLED_RPI.a
	$(CXX) -L/home/pi/mlx90640-library $^ -o $@ $(I2C_LIBS)  $(LDLIBS)

$(BUILD_DIR)testRihem: $(SRC_DIR)testRihem.o libMLX90640_API.a /usr/local/lib/libSSD1306_OLED_RPI.a
	$(CXX) -L/home/pi/mlx90640-library $^ -o $@ $(I2C_LIBS) $(LDLIBS)

$(BUILD_DIR)claude: $(SRC_DIR)claude.o libMLX90640_API.a
	$(CXX) -L/home/pi/mlx90640-library $^ -o $@ $(I2C_LIBS)

$(BUILD_DIR)GUI: $(SRC_DIR)GUI.o libMLX90640_API.a 
	$(CXX) -L/home/pi/mlx90640-library $^ -o $@ $(I2C_LIBS) $(LDLIBS)

$(BUILD_DIR)oled: $(SRC_DIR)oled.o libMLX90640_API.a /usr/local/lib/libSSD1306_OLED_RPI.a
	$(CXX) -L/home/pi/mlx90640-library $^ -o $@ $(I2C_LIBS) $(LDLIBS)

$(BUILD_DIR)testoled: $(SRC_DIR)testoled.o libMLX90640_API.a /usr/local/lib/libSSD1306_OLED_RPI.a
	$(CXX) -L/home/pi/mlx90640-library $^ -o $@ $(I2C_LIBS) $(LDLIBS)

$(BUILD_DIR)testInterpolation: $(SRC_DIR)testInterpolation.o libMLX90640_API.a
	$(CXX) -L/home/pi/mlx90640-library $^ -o $@ $(I2C_LIBS)

$(BUILD_DIR)rawrgb: $(SRC_DIR)rawrgb.o libMLX90640_API.a
	$(CXX) -L/home/pi/mlx90640-library $^ -o $@ $(I2C_LIBS)

$(BUILD_DIR)step: $(SRC_DIR)step.o libMLX90640_API.a
	$(CXX) -L/home/pi/mlx90640-library $^ -o $@ $(I2C_LIBS)

$(BUILD_DIR)fbuf: $(SRC_DIR)fbuf.o $(LIB_DIR)fb.o libMLX90640_API.a
	$(CXX) -L/home/pi/mlx90640-library $^ -o $@ $(I2C_LIBS)

$(BUILD_DIR)interp: $(SRC_DIR)interp.o $(LIB_DIR)interpolate.o $(LIB_DIR)fb.o libMLX90640_API.a
	$(CXX) -L/home/pi/mlx90640-library $^ -o $@ $(I2C_LIBS)

$(BUILD_DIR)video: $(SRC_DIR)video.o $(LIB_DIR)fb.o libMLX90640_API.a
	$(CXX) -L/home/pi/mlx90640-library $^ -o $@ $(I2C_LIBS) -lavcodec -lavutil -lavformat

clean:
	rm -f $(examples_output)
	rm -f $(SRC_DIR)*.o
	rm -f $(LIB_DIR)*.o
	rm -f functions/*.o
	rm -f *.o
	rm -f *.so
	rm -f *.a

install: libMLX90640_API.a libMLX90640_API.so
	install -d $(DESTDIR)$(PREFIX)/lib/
	install -m 644 libMLX90640_API.a $(DESTDIR)$(PREFIX)/lib/
	install -m 644 libMLX90640_API.so $(DESTDIR)$(PREFIX)/lib/
	install -d $(DESTDIR)$(PREFIX)/	include/MLX90640/
	install -m 644 headers/*.h $(DESTDIR)$(PREFIX)/include/MLX90640/
	ldconfig
