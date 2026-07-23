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
#include <SDL2/SDL.h>
#ifndef NOTF
#include <SDL2/SDL_ttf.h>
#endif
#include <SDL2/SDL_ttf.h>
#include <bcm2835.h>
#include "headers/MLX90640_API.h"

// ── sensor constants ────────────────────────────────────────────────────────
#define MLX_I2C_ADDR 0x33
#define SRC_W 32
#define SRC_H 24
#define SCALE 3                 // internal upscale (keeps the blur)
#define DST_W (SRC_W * SCALE)  // 96
#define DST_H (SRC_H * SCALE)  // 72

// ── window constants ─────────────────────────────────────────────────────────
static int WIN_SCALE = 5;       // each upscaled pixel → 8×8 screen pixels
                                 // change with + / - keys at runtime

// ── colour palettes (LUT, 256 entries each) ─────────────────────────────────
struct RGB { uint8_t r, g, b; };

static void buildLut(int palette, RGB lut[256])
{
    struct Stop { float t; uint8_t r, g, b; };
    static const Stop iron[] = {
        {0.00f,  0,   0,   0},
        {0.20f, 30,   0,  80},
        {0.45f,120,   0, 120},
        {0.65f,200,  30,   0},
        {0.80f,255, 120,   0},
        {0.92f,255, 220,   0},
        {1.00f,255, 255, 200},
    };
    const Stop* stops = iron;
    int nStops = 7;

    for (int i = 0; i < 256; i++) {
        float t = i / 255.0f;
        int lo = 0;
        for (int s = 0; s < nStops - 1; s++)
            if (t >= stops[s].t) lo = s;
        int hi = std::min(lo + 1, nStops - 1);
        float f = (t - stops[lo].t) / std::max(1e-6f, stops[hi].t - stops[lo].t);
        lut[i].r = (uint8_t)(stops[lo].r + (stops[hi].r - stops[lo].r) * f);
        lut[i].g = (uint8_t)(stops[lo].g + (stops[hi].g - stops[lo].g) * f);
        lut[i].b = (uint8_t)(stops[lo].b + (stops[hi].b - stops[lo].b) * f);
    }
}

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

// ── overlay text helper (no-op when NOTF) ────────────────────────────────────
#ifndef NOTF
static void drawText(SDL_Renderer* ren, TTF_Font* font,
                     const char* text, int x, int y,
                     SDL_Color fg)
{
    if (!font) return;
    SDL_Surface* surf = TTF_RenderText_Blended(font, text, fg);
    if (!surf) return;
    SDL_Texture* tex = SDL_CreateTextureFromSurface(ren, surf);
    SDL_Rect dst { x, y, surf->w, surf->h };
    SDL_RenderCopy(ren, tex, nullptr, &dst);
    SDL_DestroyTexture(tex);
    SDL_FreeSurface(surf);
}
#endif

// ── main ─────────────────────────────────────────────────────────────────────
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

    // ── SDL init ─────────────────────────────────────────────────────────────
    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        fprintf(stderr, "SDL_Init error: %s\n", SDL_GetError());
        return 1;
    }

#ifndef NOTF
    TTF_Init();
    // Try common monospaced fonts – falls back gracefully if none found
    TTF_Font* font = nullptr;
    const char* fontPaths[] = {
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeMono.ttf",
        nullptr
    };
    for (int i = 0; fontPaths[i]; i++) {
        font = TTF_OpenFont(fontPaths[i], 14);
        if (font) break;
    }
#endif

    int winW = DST_W * WIN_SCALE;
    int winH = DST_H * WIN_SCALE + 30;   

    SDL_Window* win = SDL_CreateWindow(
        "MLX90640 Thermal Viewer",
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
        winW, winH,
        SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE
    );
    SDL_Renderer* ren = SDL_CreateRenderer(
        win, -1,
        SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC
    );
    SDL_SetHint(SDL_HINT_RENDER_SCALE_QUALITY, "0");  // crisp nearest-neighbour

    // Texture: DST_W × DST_H pixels, updated every frame
    SDL_Texture* tex = SDL_CreateTexture(
        ren,
        SDL_PIXELFORMAT_RGB24,
        SDL_TEXTUREACCESS_STREAMING,
        DST_W, DST_H
    );

    // ── colour palette & LUT ─────────────────────────────────────────────────
    int paletteIdx = 0;
    RGB lut[256];
    buildLut(paletteIdx, lut);

    const char* paletteNames[] = {"iron","inferno","hot","cool","gray"};

    // ── working buffers ───────────────────────────────────────────────────────
    static float oriented[SRC_H][SRC_W];
    static float upscaled[DST_H][DST_W];
    static uint8_t pixels[DST_H * DST_W * 3];   // RGB for texture upload

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

        // ── events ────────────────────────────────────────────────────────────
        SDL_Event ev;
        while (SDL_PollEvent(&ev)) {
            if (ev.type == SDL_QUIT) { running = false; break; }
            if (ev.type == SDL_KEYDOWN) {
                switch (ev.key.keysym.sym) {
                    case SDLK_ESCAPE: case SDLK_q: running = false; break;
                    case SDLK_p:  paused = !paused; break;
                    case SDLK_1: case SDLK_2: case SDLK_3:
                    case SDLK_4: case SDLK_5:
                        paletteIdx = ev.key.keysym.sym - SDLK_1;
                        buildLut(paletteIdx, lut);
                        printf("palette activated : %d",paletteIdx);
                        break;
                    case SDLK_EQUALS: case SDLK_PLUS:
                        WIN_SCALE = std::min(16, WIN_SCALE + 1);
                        SDL_SetWindowSize(win,
                            DST_W * WIN_SCALE,
                            DST_H * WIN_SCALE + 28);
                        break;
                    case SDLK_MINUS:
                        WIN_SCALE = std::max(2, WIN_SCALE - 1);
                        SDL_SetWindowSize(win,
                            DST_W * WIN_SCALE,
                            DST_H * WIN_SCALE + 28);
                        break;
                }
            }
            if (ev.type == SDL_MOUSEMOTION) {
                mouseX = ev.motion.x;
                mouseY = ev.motion.y;
            }
        }

        if (!running) break;

        // ── sensor read (skipped when paused) ─────────────────────────────────
        if (!paused) {
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
            
	    std::ofstream file("./data/CameraArray.txt");
            
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

        // ── render ─────────────────────────────────────────────────────────────
        SDL_GetWindowSize(win, &winW, &winH);
        int imgH = winH - 28;

        SDL_SetRenderDrawColor(ren, 10, 14, 26, 255);
        SDL_RenderClear(ren);

        // thermal image – scale to fill window width, keep aspect
        SDL_Rect dst { 0, 0, winW, imgH };
        SDL_RenderCopy(ren, tex, nullptr, &dst);

        // crosshair
        if (mouseX >= 0 && mouseY >= 0 && mouseY < imgH) {
            SDL_SetRenderDrawColor(ren, 255, 255, 255, 120);
            SDL_RenderDrawLine(ren, mouseX, 0, mouseX, imgH);
            SDL_RenderDrawLine(ren, 0, mouseY, winW, mouseY);
        }

        // status bar background
        SDL_Rect bar { 0, imgH, winW, 28 };
        SDL_SetRenderDrawColor(ren, 17, 24, 39, 255);
        SDL_RenderFillRect(ren, &bar);

        // thin separator line
        SDL_SetRenderDrawColor(ren, 40, 60, 80, 255);
        SDL_RenderDrawLine(ren, 0, imgH, winW, imgH);

#ifndef NOTF
        if (font) {
            SDL_Color white = {220, 230, 240, 255};
            SDL_Color hot   = {255, 120,  40, 255};
            SDL_Color cold  = { 80, 200, 255, 255};
            SDL_Color dim   = {100, 120, 140, 255};

            char buf[128];

            snprintf(buf, sizeof(buf), "MAX %.1f C", maxT);
            drawText(ren, font, buf, 8, imgH + 6, hot);

            snprintf(buf, sizeof(buf), "MIN %.1f C", minT);
            drawText(ren, font, buf, 130, imgH + 6, cold);

            snprintf(buf, sizeof(buf), "AVG %.1f C", avgT);
            drawText(ren, font, buf, 250, imgH + 6, white);

            snprintf(buf, sizeof(buf), "%.1f fps  frame %d  sub %d  palette:%s",
                     fps, frameN, subpage, paletteNames[paletteIdx % 5]);
            drawText(ren, font, buf, 390, imgH + 6, dim);

            // cursor pixel temp
            if (mouseX >= 0 && mouseY >= 0 && mouseY < imgH) {
                int px = (int)((float)mouseX / winW  * DST_W);
                int py = (int)((float)mouseY / imgH  * DST_H);
                px = std::max(0, std::min(DST_W-1, px));
                py = std::max(0, std::min(DST_H-1, py));
                float val = upscaled[py][px];
                snprintf(buf, sizeof(buf), "cursor %.1f C (%d,%d)",
                         val, px/SCALE, py/SCALE);
                drawText(ren, font, buf, winW - 220, imgH + 6, white);
            }
        }
#endif

        SDL_RenderPresent(ren);
    }

    // ── cleanup ───────────────────────────────────────────────────────────────
	bcm2835_i2c_end();

    SDL_DestroyTexture(tex);

    SDL_DestroyRenderer(ren);
    SDL_DestroyWindow(win);
#ifndef NOTF
    if (font) TTF_CloseFont(font);
    TTF_Quit();
#endif
    SDL_Quit();
    return 0;
}
