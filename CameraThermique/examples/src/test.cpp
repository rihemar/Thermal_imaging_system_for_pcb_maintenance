// =============================================================================
//  thermal_viewer.cpp
//  MLX90640 thermal camera viewer with SDL2 display and SSD1306 OLED output
// =============================================================================

#include <stdint.h>
#include <iostream>
#include <cstring>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <csignal>
#include <atomic>

#include <SDL2/SDL.h>
#include <SDL2/SDL_ttf.h>

#include "SSD1306_OLED.hpp"
#include "headers/MLX90640_API.h"

// =============================================================================
//  Constants
// =============================================================================

// Sensor I2C address
#define MLX_I2C_ADDR 0x33

// Raw sensor resolution
#define SRC_W 32
#define SRC_H 24

// Bilinear upscale factor (keeps blur smoothness)
#define SCALE 3
#define DST_W (SRC_W * SCALE)   // 96
#define DST_H (SRC_H * SCALE)   // 72

// OLED display dimensions  (must be exact multiples of SRC_W / SRC_H)
// 128 / 32 = 4,  64 / 24 ≈ 2.67 — NOT an integer, so we fit to width (×4)
// and centre vertically, leaving 64 - 24*4/4... actually 128/32=4, 64/24=2.67
// Simplest correct approach: scale ×4 horizontally, ×(64/24) is non-integer,
// so we use nearest-neighbour sampling: for each OLED pixel (ox, oy) we sample
// oriented[ oy * SRC_H / OLED_H ][ ox * SRC_W / OLED_W ]  (integer division).
// This stretches the image to fill 128×64 without gaps or overflow.
#define OLED_W 128
#define OLED_H 64

// Default window zoom (each upscaled pixel → N×N screen pixels); +/- to change
static int WIN_SCALE = 10;

// =============================================================================
//  Types
// =============================================================================

struct RGB { uint8_t r, g, b; };

// =============================================================================
//  Colour palette builder
//  Generates a 256-entry RGB LUT by linearly interpolating between colour stops.
//  palette 0=iron  1=inferno  2=hot  3=cool  4=gray
// =============================================================================

static void buildLut(int palette, RGB lut[256])
{
    struct Stop { float t; uint8_t r, g, b; };

    static const Stop iron[] = {
        {0.00f,   0,   0,   0},
        {0.20f,  30,   0,  80},
        {0.45f, 120,   0, 120},
        {0.65f, 200,  30,   0},
        {0.80f, 255, 120,   0},
        {0.92f, 255, 220,   0},
        {1.00f, 255, 255, 200},
    };
    static const Stop inferno[] = {
        {0.00f,   0,   0,   4},
        {0.20f,  50,   5,  90},
        {0.40f, 115,  25, 140},
        {0.60f, 180,  60,  80},
        {0.80f, 230, 120,  30},
        {0.92f, 250, 200,  30},
        {1.00f, 250, 250, 150},
    };
    static const Stop hot[] = {
        {0.00f,   0,   0,   0},
        {0.33f, 180,   0,   0},
        {0.60f, 255,  80,   0},
        {0.80f, 255, 200,   0},
        {1.00f, 255, 255, 255},
    };
    static const Stop cool[] = {
        {0.00f,  80,   0, 120},
        {0.25f,   0,  40, 180},
        {0.50f,   0, 160, 220},
        {0.75f,   0, 220, 180},
        {1.00f, 100, 250, 100},
    };
    static const Stop gray[] = {
        {0.00f,   0,   0,   0},
        {1.00f, 255, 255, 255},
    };

    const Stop* stops = iron;
    int nStops = 7;
    switch (palette % 5) {
        case 0: stops = iron;    nStops = 7; break;
        case 1: stops = inferno; nStops = 7; break;
        case 2: stops = hot;     nStops = 5; break;
        case 3: stops = cool;    nStops = 5; break;
        case 4: stops = gray;    nStops = 2; break;
    }

    for (int i = 0; i < 256; i++) {
        float t  = i / 255.0f;
        int   lo = 0;
        for (int s = 0; s < nStops - 1; s++)
            if (t >= stops[s].t) lo = s;
        int   hi = std::min(lo + 1, nStops - 1);
        float f  = (t - stops[lo].t) /
                   std::max(1e-6f, stops[hi].t - stops[lo].t);

        lut[i].r = (uint8_t)(stops[lo].r + (stops[hi].r - stops[lo].r) * f);
        lut[i].g = (uint8_t)(stops[lo].g + (stops[hi].g - stops[lo].g) * f);
        lut[i].b = (uint8_t)(stops[lo].b + (stops[hi].b - stops[lo].b) * f);
    }
}

// =============================================================================
//  Bilinear-style upscale  (SRC → DST via 3×3 neighbourhood blending)
//  Each source pixel expands to a SCALE×SCALE block whose corners/edges
//  are blended with their cardinal and diagonal neighbours.
// =============================================================================

static void slidingWindowUpscale(const float src[SRC_H][SRC_W],
                                        float dst[DST_H][DST_W])
{
    for (int r = 0; r < SRC_H; r++) {
        const int rm = (r > 0)          ? r - 1 : 0;
        const int rp = (r < SRC_H - 1) ? r + 1 : r;

        for (int c = 0; c < SRC_W; c++) {
            const int cm = (c > 0)          ? c - 1 : 0;
            const int cp = (c < SRC_W - 1) ? c + 1 : c;

            const float ce = src[r][c];
            const float n  = src[rm][c],  s  = src[rp][c];
            const float w  = src[r][cm],  e  = src[r][cp];
            const float nw = src[rm][cm], ne = src[rm][cp];
            const float sw = src[rp][cm], se = src[rp][cp];

            // 3×3 sub-block — each position blends its surrounding samples
            const float block[SCALE][SCALE] = {
                { (nw + n + w + ce) / 4.0f,  (ce + n)              / 2.0f,  (ce + n + e + ne) / 4.0f },
                { (ce + w)          / 2.0f,   ce,                            (ce + e)          / 2.0f },
                { (ce + s + w + sw) / 4.0f,  (ce + s)              / 2.0f,  (ce + s + e + se) / 4.0f },
            };

            const int baseR = r * SCALE;
            const int baseC = c * SCALE;
            for (int br = 0; br < SCALE; br++)
                for (int bc = 0; bc < SCALE; bc++)
                    dst[baseR + br][baseC + bc] = block[br][bc];
        }
    }
}

// =============================================================================
//  OLED upscale  32×24 → 128×64  (nearest-neighbour, then threshold to 1-bit)
//
//  The SSD1306 OLEDBitmap() expects a packed 1-bit buffer laid out in
//  horizontal pages of 8 rows each (standard SSD1306 page addressing):
//    byte index = (row / 8) * OLED_W + col
//    bit  index = row % 8          (LSB = topmost row of the page)
//
//  For each OLED pixel (col, row) we:
//    1. Map back to the nearest source pixel via integer scaling.
//    2. Threshold the temperature: pixels above the midpoint → white (1).
//    3. Pack the bit into the correct byte.
// =============================================================================

static void oledUpscale(const float src[SRC_H][SRC_W],
                         uint8_t    dst[OLED_H * OLED_W / 8],
                         float      minVal, float maxVal)
{
    memset(dst, 0, OLED_H * OLED_W / 8);

    const float mid   = (minVal + maxVal) * 0.5f;

    for (int oy = 0; oy < OLED_H; oy++) {
        // nearest-neighbour row mapping: 0..OLED_H-1 → 0..SRC_H-1
        const int sy = oy * SRC_H / OLED_H;

        for (int ox = 0; ox < OLED_W; ox++) {
            // nearest-neighbour col mapping: 0..OLED_W-1 → 0..SRC_W-1
            const int sx = ox * SRC_W / OLED_W;

            // Pixel is ON (white) when temperature is above the midpoint
            if (src[sy][sx] >= mid) {
                const int byteIdx = (oy / 8) * OLED_W + ox;
                const int bitIdx  = (oy % 8);          // LSB = bottom of page
                dst[byteIdx] |= (1u << bitIdx);
            }
        }
    }
}

static void oledUpscale2(const float src[SRC_H][SRC_W],
                         uint8_t    dst[OLED_H * OLED_W],
                         float      minVal, float maxVal)
{
    memset(dst, 0, OLED_H * OLED_W);

    const float mid   = (minVal + maxVal) * 0.5f;

    for (int oy = 0; oy < OLED_H; oy++) {
        // nearest-neighbour row mapping: 0..OLED_H-1 → 0..SRC_H-1
        const int sy = oy * SRC_H / OLED_H;

        for (int ox = 0; ox < OLED_W; ox++) {
            // nearest-neighbour col mapping: 0..OLED_W-1 → 0..SRC_W-1
            const int sx = ox * SRC_W / OLED_W;

            // Pixel is ON (white) when temperature is above the midpoint
            if (src[sy][sx] >= mid) {
                const int byteIdx = (oy) * OLED_W + ox;
                dst[byteIdx] = 0xfe;
            }
        }
    }
}

// =============================================================================
//  SDL_ttf text helper  (compiled out when NOTF is defined)
// =============================================================================

#ifndef NOTF
static void drawText(SDL_Renderer* ren, TTF_Font* font,
                     const char* text, int x, int y, SDL_Color fg)
{
    if (!font) return;
    SDL_Surface* surf = TTF_RenderText_Blended(font, text, fg);
    if (!surf)  return;
    SDL_Texture* tex = SDL_CreateTextureFromSurface(ren, surf);
    if (tex) {
        SDL_Rect dst { x, y, surf->w, surf->h };
        SDL_RenderCopy(ren, tex, nullptr, &dst);
        SDL_DestroyTexture(tex);
    }
    SDL_FreeSurface(surf);
}
#endif

// =============================================================================
//  Static / global buffers
//  (static keeps them in BSS; avoids large stack allocations)
// =============================================================================

static SSD1306  myOLED(OLED_W, OLED_H);

// OLED 1-bit framebuffer — packed, page-addressed: (OLED_H/8) pages × OLED_W bytes
// Total = 128 × 64 / 8 = 1024 bytes
static uint8_t oledBuf[OLED_H * OLED_W / 8];

// Intermediate sensor buffers
static uint16_t        eeMLX90640[832];
static uint16_t        sensorFrame[834];
static float           mlx90640To[768];         // raw temperature array

// Processing buffers
static float           oriented[SRC_H][SRC_W];  // vertically-flipped sensor data
static float           upscaled[DST_H][DST_W];  // bilinear-upscaled result
static uint8_t         pixels[DST_H * DST_W * 3]; // RGB pixels for SDL texture

// signal handler for ^c ::
	std::atomic<bool>  running(true);
void signal_handler(int signal_num){
	running = false;
}


// ======================= Function space ===================
bool oledSetup()
{

        const uint16_t I2C_Speed = BCM2835_I2C_CLOCK_DIVIDER_626; //  bcm2835I2CClockDivider enum , see readme.
        const uint8_t I2C_Address = 0x3C;
        bool I2C_debug = false;
        printf("OLED Test Begin\r\n");

        // Check if Bcm28235 lib installed and print version.
        if(!bcm2835_init())
        {
                printf("Error 1201: init bcm2835 library , Is it installed ?\r\n");
                return false;
        }

	bcm2835_i2c_begin();        
        if(!myOLED.OLED_I2C_ON())
        {
                printf("Error 1202: bcm2835_i2c_begin :Cannot start I2C, Running as root?\n");
                bcm2835_close(); // Close the library
                return false;
        }

        printf("SSD1306 library Version Number :: %u\r\n",myOLED.getLibVerNum());
        printf("bcm2835 library Version Number :: %u\r\n",bcm2835_version());
        bcm2835_delay(500);
        myOLED.OLEDbegin(I2C_Speed, I2C_Address, I2C_debug); // initialize the OLED
        myOLED.OLEDFillScreen(0xF0, 0); // splash screen bars, optional just for effect
	myOLED.OLEDSetBufferPtr(OLED_W, OLED_H, oledBuf, sizeof(oledBuf));
	bcm2835_delay(1000);
        return true;
}

void oledCleanup()
{
        myOLED.OLEDPowerDown(); //Switch off display
        myOLED.OLED_I2C_OFF(); // Switch off I2C , optional may effect other programs & devices
        bcm2835_close(); // Close the bcm2835 library
        printf("OLED Test End\r\n");
}

// =============================================================================
//  main
// =============================================================================

int main()
{
	std::signal(SIGINT , signal_handler);
    // ── BCM2835 / OLED init ───────────────────────────────────────────────────
	oledSetup() ;   

    // ── MLX90640 sensor init ──────────────────────────────────────────────────
    paramsMLX90640 mlx90640;

    // NOTE: SetDeviceMode left at default (interleaved); SubPageRepeat=0 means
    //       alternating sub-pages, which is the normal operating mode.
    MLX90640_SetSubPageRepeat(MLX_I2C_ADDR, 0);
    MLX90640_SetRefreshRate  (MLX_I2C_ADDR, 0b010);  // 2 Hz
    MLX90640_SetChessMode    (MLX_I2C_ADDR);

    MLX90640_DumpEE          (MLX_I2C_ADDR, eeMLX90640);
    MLX90640_ExtractParameters(eeMLX90640, &mlx90640);

    const float emissivity = 0.95f;

    // ── SDL2 init ─────────────────────────────────────────────────────────────
    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        fprintf(stderr, "SDL_Init error: %s\n", SDL_GetError());
        return 1;
    }

#ifndef NOTF
    TTF_Init();
    TTF_Font* font = nullptr;
    static const char* fontPaths[] = {
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
    if (!win) {
        fprintf(stderr, "SDL_CreateWindow error: %s\n", SDL_GetError());
                SDL_Quit();
                return 1;
            }

    SDL_Renderer* ren = SDL_CreateRenderer(
        win, -1,
        SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC
    );
    if (!ren) {
        fprintf(stderr, "SDL_CreateRenderer error: %s\n", SDL_GetError());
        SDL_DestroyWindow(win);
        SDL_Quit();
        return 1;
    }

    // Nearest-neighbour scaling → crisp pixels, no blur from SDL
    SDL_SetHint(SDL_HINT_RENDER_SCALE_QUALITY, "0");

    SDL_Texture* tex = SDL_CreateTexture(
        ren,
        SDL_PIXELFORMAT_RGB24,
        SDL_TEXTUREACCESS_STREAMING,
        DST_W, DST_H
    );
    if (!tex) {
        fprintf(stderr, "SDL_CreateTexture error: %s\n", SDL_GetError());
        SDL_DestroyRenderer(ren);
        SDL_DestroyWindow(win);
        SDL_Quit();
        return 1;
    }

    // ── Palette ───────────────────────────────────────────────────────────────
    static const char* paletteNames[] = {"iron", "inferno", "hot", "cool", "gray"};
    int  paletteIdx = 0;
    RGB  lut[256];
    buildLut(paletteIdx, lut);

    // ── State ─────────────────────────────────────────────────────────────────
    bool  paused  = false;
    int   subpage = 0;
    int   frameN  = 0;

    float minT = 20.f, maxT = 50.f, avgT = 35.f;
    float eTa  = 0.f;
    float fps  = 0.f;

    int mouseX = -1, mouseY = -1;

    auto tLast = std::chrono::steady_clock::now();

    // ── Main loop ─────────────────────────────────────────────────────────────
    while (running) {

        // ── Event handling ────────────────────────────────────────────────────
        SDL_Event ev;
        while (SDL_PollEvent(&ev)) {
            if (ev.type == SDL_QUIT) {
                running = false;
                break;
            }
            if (ev.type == SDL_KEYDOWN) {
                switch (ev.key.keysym.sym) {
                    case SDLK_ESCAPE:
                    case SDLK_q:
                        running = false;
                        break;

                    case SDLK_p:
                        paused = !paused;
                        printf("Paused: %s\n", paused ? "yes" : "no");
                        break;

                    // Number keys 1-5 select palette
                    case SDLK_1: case SDLK_2: case SDLK_3:
                    case SDLK_4: case SDLK_5:
                        paletteIdx = ev.key.keysym.sym - SDLK_1;
                        buildLut(paletteIdx, lut);
                        printf("Palette: %s\n", paletteNames[paletteIdx]);
                        break;

                    // Zoom in/out
                    case SDLK_EQUALS:
                    case SDLK_PLUS:
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

        // ── Sensor acquisition (skipped while paused) ─────────────────────────
        if (!paused) {
            int status = MLX90640_GetFrameData(MLX_I2C_ADDR, sensorFrame);
            if (status < 0) {
                printf("Frame read failed: %d\n", status);
                continue;
            }

            eTa     = MLX90640_GetTa(sensorFrame, &mlx90640);
            subpage = MLX90640_GetSubPageNumber(sensorFrame);

            MLX90640_CalculateTo(sensorFrame, &mlx90640, emissivity, eTa, mlx90640To);

            // Correct broken and outlier pixels
            MLX90640_BadPixelsCorrection(mlx90640.brokenPixels,  mlx90640To, 1, &mlx90640);
            MLX90640_BadPixelsCorrection(mlx90640.outlierPixels, mlx90640To, 1, &mlx90640);

            // Sensor is mounted upside-down → flip vertically
            for (int r = 0; r < SRC_H; r++)
                for (int c = 0; c < SRC_W; c++)
                    oriented[r][c] = mlx90640To[SRC_W * (SRC_H - 1 - r) + c];

            // ── OLED output ───────────────────────────────────────────────────
            // Scale 32×24 → 128×64 with nearest-neighbour, threshold to 1-bit,
            // pack into the SSD1306 page-addressed format, then blit.
            // minT/maxT are computed after upscaling below, so on the very first
            // frame they hold their initial values (20 / 50 °C) — acceptable.
            oledUpscale(oriented, oledBuf, minT, maxT);
            myOLED.OLEDBitmap(0, 0, OLED_W, OLED_H, oledBuf, false);
            myOLED.OLEDupdate();

            // ── Upscale ───────────────────────────────────────────────────────
            slidingWindowUpscale(oriented, upscaled);

            // ── Statistics ────────────────────────────────────────────────────
            minT = 300.f; maxT = -300.f;
            double sum = 0.0;

            for (int i = 0; i < DST_H * DST_W; i++) {
                float v = upscaled[i / DST_W][i % DST_W];
                if (v < minT) minT = v;
                if (v > maxT) maxT = v;
                sum += v;
            }
            avgT = static_cast<float>(sum / (DST_H * DST_W));

            // ── Map temperatures to RGB via LUT ───────────────────────────────
            float range = maxT - minT;
            if (range < 0.01f) range = 0.01f;

            for (int i = 0; i < DST_H * DST_W; i++) {
                int n = static_cast<int>(
                    ((upscaled[i / DST_W][i % DST_W] - minT) / range) * 255.f);
                n = std::max(0, std::min(255, n));
                pixels[i * 3    ] = lut[n].r;
                pixels[i * 3 + 1] = lut[n].g;
                pixels[i * 3 + 2] = lut[n].b;
            }

            SDL_UpdateTexture(tex, nullptr, pixels, DST_W * 3);

            // ── FPS (exponential moving average) ─────────────────────────────
            auto  tNow = std::chrono::steady_clock::now();
            float dt   = std::chrono::duration<float>(tNow - tLast).count();
            tLast = tNow;
            fps   = 0.9f * fps + 0.1f * (1.f / dt);

            frameN++;
        }

        // ── Rendering ─────────────────────────────────────────────────────────
        SDL_GetWindowSize(win, &winW, &winH);
        const int imgH = winH - 28;   // reserve bottom 28 px for status bar

        SDL_SetRenderDrawColor(ren, 10, 14, 26, 255);
        SDL_RenderClear(ren);

        // Thermal image — fill window width, preserve aspect
        SDL_Rect imgRect { 0, 0, winW, imgH };
        SDL_RenderCopy(ren, tex, nullptr, &imgRect);

        // Crosshair at cursor position
        if (mouseX >= 0 && mouseY >= 0 && mouseY < imgH) {
            SDL_SetRenderDrawColor(ren, 255, 255, 255, 120);
            SDL_RenderDrawLine(ren, mouseX, 0,     mouseX, imgH);
            SDL_RenderDrawLine(ren, 0,      mouseY, winW,  mouseY);
        }

        // Status bar background
        SDL_Rect bar { 0, imgH, winW, 28 };
        SDL_SetRenderDrawColor(ren, 17, 24, 39, 255);
        SDL_RenderFillRect(ren, &bar);

        // Status bar separator line
        SDL_SetRenderDrawColor(ren, 40, 60, 80, 255);
        SDL_RenderDrawLine(ren, 0, imgH, winW, imgH);

#ifndef NOTF
        if (font) {
            const SDL_Color white = {220, 230, 240, 255};
            const SDL_Color hot_c = {255, 120,  40, 255};
            const SDL_Color cold  = { 80, 200, 255, 255};
            const SDL_Color dim   = {100, 120, 140, 255};

            char buf[128];

            snprintf(buf, sizeof(buf), "MAX %.1f C", maxT);
            drawText(ren, font, buf, 8, imgH + 6, hot_c);

            snprintf(buf, sizeof(buf), "MIN %.1f C", minT);
            drawText(ren, font, buf, 130, imgH + 6, cold);

            snprintf(buf, sizeof(buf), "AVG %.1f C", avgT);
            drawText(ren, font, buf, 250, imgH + 6, white);

            snprintf(buf, sizeof(buf), "%.1f fps  frame %d  sub %d  palette: %s",
                     fps, frameN, subpage, paletteNames[paletteIdx % 5]);
            drawText(ren, font, buf, 390, imgH + 6, dim);

            // Cursor temperature readout
            if (mouseX >= 0 && mouseY >= 0 && mouseY < imgH) {
                int px = static_cast<int>((float)mouseX / winW * DST_W);
                int py = static_cast<int>((float)mouseY / imgH * DST_H);
                px = std::max(0, std::min(DST_W - 1, px));
                py = std::max(0, std::min(DST_H - 1, py));

                snprintf(buf, sizeof(buf), "cursor %.1f C  (%d,%d)",
                         upscaled[py][px], px / SCALE, py / SCALE);
                drawText(ren, font, buf, winW - 220, imgH + 6, white);
            }
        }
#endif

        SDL_RenderPresent(ren);
    }

    // ── Cleanup ───────────────────────────────────────────────────────────────
    oledCleanup();
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
