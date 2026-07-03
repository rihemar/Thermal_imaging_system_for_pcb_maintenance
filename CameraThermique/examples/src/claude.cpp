#include <stdint.h>
#include <iostream>
#include <cstring>
#include <fstream>
#include <chrono>
#include <thread>
#include "headers/MLX90640_API.h"
#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_YELLOW  "\x1b[33m"
#define ANSI_COLOR_BLUE    "\033[38;2;0;0;139m"
#define ANSI_COLOR_ORANGE  "\033[38;2;255;165;0m"
#define ANSI_COLOR_MAGENTA "\x1b[35m"
#define ANSI_COLOR_CYAN    "\x1b[36m"
#define ANSI_COLOR_NONE    "\x1b[30m"
#define ANSI_COLOR_RESET   "\x1b[0m"
#define FMT_STRING "\u2588\u2588"
#define MLX_I2C_ADDR 0x33

// Sensor grid is 32 columns x 24 rows. We scale it up x3 to 96x72.
#define SRC_W 32
#define SRC_H 24
#define SCALE 3
#define DST_W (SRC_W * SCALE)
#define DST_H (SRC_H * SCALE)

// Takes the 32x24 oriented temperature grid and produces a 96x72 grid.
// For every source pixel, a 3x3 block of 9 output pixels is generated:
// the center of the block keeps the original value untouched, and the
// 8 pixels around it are blended with the matching neighbor(s) "on that
// side" (edge neighbor for the 4 mid pixels, edge+corner neighbors for
// the 4 corner pixels). This scales and smooths the image in one pass,
// instead of doing a blocky resize and then a separate blur.
void slidingWindowUpscale(const float src[SRC_H][SRC_W], float dst[DST_H][DST_W]) {
    for (int r = 0; r < SRC_H; r++) {
        int rm = (r > 0) ? r - 1 : 0;          // row above, clamped at top edge
        int rp = (r < SRC_H - 1) ? r + 1 : r;  // row below, clamped at bottom edge
        for (int c = 0; c < SRC_W; c++) {
            int cm = (c > 0) ? c - 1 : 0;          // column to the left, clamped
            int cp = (c < SRC_W - 1) ? c + 1 : c;  // column to the right, clamped

            float center = src[r][c];
            float n  = src[rm][c];
            float s  = src[rp][c];
            float w  = src[r][cm];
            float e  = src[r][cp];
            float nw = src[rm][cm];
            float ne = src[rm][cp];
            float sw = src[rp][cm];
            float se = src[rp][cp];

            float block[SCALE][SCALE] = {
                { (center + n + w + nw) / 4.0f,  (center + n) / 2.0f,  (center + n + e + ne) / 4.0f },
                { (center + w) / 2.0f,            center,               (center + e) / 2.0f          },
                { (center + s + w + sw) / 4.0f,  (center + s) / 2.0f,  (center + s + e + se) / 4.0f }
            };

            int baseR = r * SCALE;
            int baseC = c * SCALE;
            for (int br = 0; br < SCALE; br++)
                for (int bc = 0; bc < SCALE; bc++)
                    dst[baseR + br][baseC + bc] = block[br][bc];
        }
    }
}

// Same discrete color buckets as before, just factored out so it can be
// called once per upscaled pixel.
void printHeatPixel(float val) {
    if (val > 99.99f) val = 99.99f;
    if (val > 53.0f) {
        printf(ANSI_COLOR_RED FMT_STRING ANSI_COLOR_RESET);
    } else if (val > 48.0f) {
        printf(ANSI_COLOR_ORANGE FMT_STRING ANSI_COLOR_RESET);
    } else if (val > 42.0f) {
        printf(ANSI_COLOR_YELLOW FMT_STRING ANSI_COLOR_RESET);
    } else if (val > 36.0f) {
        printf(ANSI_COLOR_CYAN FMT_STRING ANSI_COLOR_RESET);
    } else if (val > 26.0f){
        printf(ANSI_COLOR_BLUE FMT_STRING ANSI_COLOR_RESET);
    }else{
        printf(ANSI_COLOR_NONE FMT_STRING ANSI_COLOR_RESET);
    }
}

int main(){
    int state = 0;
    printf("Starting...\n");
    static uint16_t eeMLX90640[832];
    float emissivity = 1;
    uint16_t frame[834];
    float eTa;
    std::fstream fs;
    MLX90640_SetDeviceMode(MLX_I2C_ADDR, 0);
    MLX90640_SetSubPageRepeat(MLX_I2C_ADDR, 0);
    MLX90640_SetRefreshRate(MLX_I2C_ADDR, 0b010);
    MLX90640_SetChessMode(MLX_I2C_ADDR);
    //MLX90640_SetSubPage(MLX_I2C_ADDR, 0);
    printf("Configured...\n");
    paramsMLX90640 mlx90640;
    MLX90640_DumpEE(MLX_I2C_ADDR, eeMLX90640);
    MLX90640_ExtractParameters(eeMLX90640, &mlx90640);
    int refresh = MLX90640_GetRefreshRate(MLX_I2C_ADDR);
    (void)refresh;
    printf("EE Dumped...\n");
    int subpage;
    static float mlx90640To[768];

    static float oriented[SRC_H][SRC_W];
    static float upscaled[DST_H][DST_W];

    while (1){
        state = !state;
        //printf("State: %d \n", state);

        // --- sensor query / frame retrieval: unchanged ---
        MLX90640_GetFrameData(MLX_I2C_ADDR, frame);
        // MLX90640_InterpolateOutliers(frame, eeMLX90640);
        eTa = MLX90640_GetTa(frame, &mlx90640);
        subpage = MLX90640_GetSubPageNumber(frame);
        MLX90640_CalculateTo(frame, &mlx90640, emissivity, eTa, mlx90640To);
        MLX90640_BadPixelsCorrection((&mlx90640)->brokenPixels, mlx90640To, 1, &mlx90640);
        MLX90640_BadPixelsCorrection((&mlx90640)->outlierPixels, mlx90640To, 1, &mlx90640);
        printf("Subpage: %d\n", subpage);
        //MLX90640_SetSubPage(MLX_I2C_ADDR,!subpage);

        // Build the 24x32 grid in display orientation (vertical flip
        // applied here, once, instead of inside the render loop).
        for (int row = 0; row < SRC_H; row++) {
            for (int col = 0; col < SRC_W; col++) {
                oriented[row][col] = mlx90640To[32 * (23 - row) + col];
            }
        }

        // Scale 32x24 -> 96x72 with the 3x3 sliding-window interpolation.
        slidingWindowUpscale(oriented, upscaled);

        // Render the upscaled grid: rows outer, columns inner, so each
        // printed line is one real row of 96 pixels.
        for (int row = 0; row < DST_H; row++) {
            for (int col = 0; col < DST_W; col++) {
                printHeatPixel(upscaled[row][col]);
            }
            std::cout << std::endl;
        }

        //std::this_thread::sleep_for(std::chrono::milliseconds(20));
        // 1 "Subpage:" line + 72 image rows = 73 lines printed per frame.
        printf("\x1b[73A");
    }
    return 0;
}
