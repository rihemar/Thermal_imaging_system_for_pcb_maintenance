#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <stdint.h>
#include <iostream>
#include <fstream>
#include <cstring>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <bcm2835.h>
#include "headers/MLX90640_API.h"

// ── sensor constants ────────────────────────────────────────────────────────
#define MLX_I2C_ADDR 0x33
#define SRC_W 32
#define SRC_H 24
#define SCALE 3                 // internal upscale (keeps the blur)
#define DST_W (SRC_W * SCALE)  // 96
#define DST_H (SRC_H * SCALE)  // 72

// ── upscale (interpolation (from 1 cell build a set of 9 cells with consideration of adjacent cells)) ─────────────────────────────────────
static void slidingWindowUpscale(const float src[SRC_H][SRC_W],
                                  float       dst[DST_H][DST_W])
{
    for (int r = 0; r < SRC_H; r++) {
        int rm = (r > 0)         ? r - 1 : 0;
        int rp = (r < SRC_H - 1) ? r + 1 : r;
        for (int c = 0; c < SRC_W; c++) {
            int cm = (c > 0)         ? c - 1 : 0;
            int cp = (c < SRC_W - 1) ? c + 1 : c;

            float ce = src[r][c];
            float n  = src[rm][c],  s  = src[rp][c];
            float w  = src[r][cm],  e  = src[r][cp];
            float nw = src[rm][cm], ne = src[rm][cp];
            float sw = src[rp][cm], se = src[rp][cp];

             float block[SCALE][SCALE] = {
                { (nw+n+w+ce)/4.0f, (ce+n)/2.f,     (ce+n+e+ne)/4.f },
                { (ce+w)/2.f,       ce,              (ce+e)/2.f      },
                { (ce+s+w+sw)/4.f, (ce+s)/2.f,     (ce+s+e+se)/4.f }
            };


            int baseR = r * SCALE, baseC = c * SCALE;
            for (int br = 0; br < SCALE; br++)
                for (int bc = 0; bc < SCALE; bc++)
                    dst[baseR+br][baseC+bc] = block[br][bc];
        }
    }
}


int main()
{
        bcm2835_i2c_begin();
    // ── sensor init (unchanged from original) ───────────────────────────────
    static uint16_t eeMLX90640[832];
    static uint16_t frame[834];
    static float    mlx90640To[768];

    float emissivity = 0.95f;
    float eTa;

    // MLX90640_SetDevieMode(MLX_I2C_ADDR, 0);
    MLX90640_SetSubPageRepeat(MLX_I2C_ADDR, 0);
    MLX90640_SetRefreshRate(MLX_I2C_ADDR, 0b010);
    MLX90640_SetChessMode(MLX_I2C_ADDR);

    paramsMLX90640 mlx90640;
    MLX90640_DumpEE(MLX_I2C_ADDR, eeMLX90640);
    MLX90640_ExtractParameters(eeMLX90640, &mlx90640);


    int winW = DST_W * WIN_SCALE;
    int winH = DST_H * WIN_SCALE + 30;

    // ── working buffers ───────────────────────────────────────────────────────
    static float oriented[SRC_H][SRC_W];
    static float upscaled[DST_H][DST_W];
    // ── state ─────────────────────────────────────────────────────────────────
    bool running  = true;
    bool paused   = false;
    int  subpage  = 0;
    int  frameN   = 0;
    float minT = 20.f, maxT = 50.f, avgT = 35.f;
    int  mouseX = -1, mouseY = -1;

    auto tLast = std::chrono::steady_clock::now();
    float fps = 0.f;

    // ── main loop ─────────────────────────────────────────────────────────────
    while (running) {
       if (!running) break;
            MLX90640_GetFrameData(MLX_I2C_ADDR, frame);
            eTa      = MLX90640_GetTa(frame, &mlx90640);
            subpage  = MLX90640_GetSubPageNumber(frame);
            MLX90640_CalculateTo(frame, &mlx90640, emissivity, eTa, mlx90640To);
            MLX90640_BadPixelsCorrection(mlx90640.brokenPixels,  mlx90640To, 1, &mlx90640);
            MLX90640_BadPixelsCorrection(mlx90640.outlierPixels, mlx90640To, 1, &mlx90640);

            // vertical flip → oriented grid
            for (int r = 0; r < SRC_H; r++)
                for (int c = 0; c < SRC_W; c++)
                    oriented[r][c] = mlx90640To[32 * (23 - r) + c];
 
          std::ofstream file("../CameraRGB/CameraArray.txt");

            if (!file.is_open()) {
                 std::cout << "Failed to open file.\n";
            return 1;
            }

            for (int i = 0; i < SRC_H; i++) {
                for (int j = 0; j < SRC_W; j++) {
                        file << oriented[i][j];
                        if (j < SRC_W - 1)
                        file << ' ';
                }
                file << '\n';
            }
            file.close();

 
            slidingWindowUpscale(oriented, upscaled);

           std::ofstream file("../CameraRGB/CameraArrayScaled.txt");

            if (!file.is_open()) {
                 std::cout << "Failed to open file.\n";
            return 1;
            }

            for (int i = 0; i < DST_H; i++) {
                for (int j = 0; j < DST_W; j++) {
                        file << upscaled[i][j];
                        if (j < DST_W - 1)
                        file << ' ';
                }
                file << '\n';
            }
            file.close();

 

            // stats
            minT = 300.0f; maxT = -300.0f;
            double sum = 0.0;
            for (int i = 0; i < DST_H * DST_W; i++) {
                float v = upscaled[i / DST_W][i % DST_W];
                if (v < minT) minT = v;
                if (v > maxT) maxT = v;
                sum += v;
            }
            avgT = (float)(sum / (DST_H * DST_W));

            // map to RGB via LUT
            float range = maxT - minT;
            if (range < 0.01f) range = 0.01f;
            for (int i = 0; i < DST_H * DST_W; i++) {
                int n = (int)(((upscaled[i / DST_W][i % DST_W] - minT) / range) * 255.f);
                n = std::max(0, std::min(255, n));
                pixels[i*3  ] = lut[n].r;
                pixels[i*3+1] = lut[n].g;
                pixels[i*3+2] = lut[n].b;
            }
            SDL_UpdateTexture(tex, nullptr, pixels, DST_W * 3);
            stbi_write_jpg("../CameraRGB/thermalCameraFrame.jpg", DST_W, DST_H, 3, pixels, 95);
            frameN++;

            // FPS
            auto tNow = std::chrono::steady_clock::now();
            float dt = std::chrono::duration<float>(tNow - tLast).count();
            tLast = tNow;
            fps = 0.9f * fps + 0.1f * (1.f / dt);
        }
    // ── cleanup ───────────────────────────────────────────────────────────────
        bcm2835_i2c_end();

    return 0;
}
