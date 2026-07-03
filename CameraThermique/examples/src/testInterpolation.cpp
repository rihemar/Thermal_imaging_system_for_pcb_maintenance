// #include <stdint.h>
// #include <iostream>
// #include <cstring>
// #include <fstream>
// #include <chrono>
// #include <thread>
// #include "headers/MLX90640_API.h"

// #define ANSI_COLOR_RED     "\x1b[31m"
// #define ANSI_COLOR_GREEN   "\x1b[32m"
// #define ANSI_COLOR_YELLOW  "\x1b[33m"
// #define ANSI_COLOR_BLUE    "\033[38;2;0;0;139m"
// #define ANSI_COLOR_ORANGE    "\033[38;2;255;165;0m"
// #define ANSI_COLOR_MAGENTA "\x1b[35m"
// #define ANSI_COLOR_CYAN    "\x1b[36m"
// #define ANSI_COLOR_NONE    "\x1b[30m"
// #define ANSI_COLOR_RESET   "\x1b[0m"

// //#define FMT_STRING "%+06.2f "
// #define FMT_STRING "\u2588\u2588"

// #define OUTPUT_W (int)(24*2)
// #define OUTPUT_H (int)(32*2)
// #define MLX_I2C_ADDR 0x33


// void put_pixel_false_colour(int x, int y, double v) {
//     // Heatmap code borrowed from: http://www.andrewnoske.com/wiki/Code_-_heatmaps_and_color_gradients
//     const int NUM_COLORS = 7;
//     static float color[NUM_COLORS][3] = { {0,0,0}, {0,0,1}, {0,1,0}, {1,1,0}, {1,0,0}, {1,0,1}, {1,1,1} };
//     int idx1, idx2;
//     float fractBetween = 0;
//     float vmin = 5.0;
//     float vmax = 50.0;
//     float vrange = vmax-vmin;
//     v -= vmin;
//     v /= vrange;
//     int ir, ig, ib;
    
//     for(int px = 0; px < IMAGE_SCALE; px++){
//         for(int py = 0; py < IMAGE_SCALE; py++){
//             if(v <= 0) {idx1=idx2=0;}
//             else if(v >= 1) {idx1=idx2=NUM_COLORS-1;}
//             else
//             {
//                 v *= (NUM_COLORS-1);
//                 idx1 = floor(v);
//                 idx2 = idx1+1;
//                 fractBetween = v - float(idx1);
//             }

//             ir = (int)((((color[idx2][0] - color[idx1][0]) * fractBetween) + color[idx1][0]) * 255.0);
//             ig = (int)((((color[idx2][1] - color[idx1][1]) * fractBetween) + color[idx1][1]) * 255.0);
//             ib = (int)((((color[idx2][2] - color[idx1][2]) * fractBetween) + color[idx1][2]) * 255.0);
//             fb_put_pixel(x + px, y + py, ir, ig, ib);
//         }
//     }
    
// }

// #define OUTPUT_W (int)(24*2)
// #define OUTPUT_H (int)(32*2)

// int main(){
//     int state = 0;
//     printf("Starting...\n");
//     static uint16_t eeMLX90640[832];
//     float emissivity = 1;
//     uint16_t frame[834];
//     float eTa;

//     std::fstream fs;

//     MLX90640_SetDeviceMode(MLX_I2C_ADDR, 0);
//     MLX90640_SetSubPageRepeat(MLX_I2C_ADDR, 0);
//     MLX90640_SetRefreshRate(MLX_I2C_ADDR, 0b010);
//     MLX90640_SetChessMode(MLX_I2C_ADDR);
//     //MLX90640_SetSubPage(MLX_I2C_ADDR, 0);
//     printf("Configured...\n");

//     paramsMLX90640 mlx90640;
//     MLX90640_DumpEE(MLX_I2C_ADDR, eeMLX90640);
//     MLX90640_ExtractParameters(eeMLX90640, &mlx90640);

//     int refresh = MLX90640_GetRefreshRate(MLX_I2C_ADDR);
//     (void)refresh;
//     printf("EE Dumped...\n");

//     int subpage;
//     static float mlx90640To[768];
//     while (1){
//         state = !state;
//         //printf("State: %d \n", state);
//         MLX90640_GetFrameData(MLX_I2C_ADDR, frame);
//         // MLX90640_InterpolateOutliers(frame, eeMLX90640);
//         eTa = MLX90640_GetTa(frame, &mlx90640);
//         subpage = MLX90640_GetSubPageNumber(frame);
//         MLX90640_CalculateTo(frame, &mlx90640, emissivity, eTa, mlx90640To);

//         MLX90640_BadPixelsCorrection((&mlx90640)->brokenPixels, mlx90640To, 1, &mlx90640);
//         MLX90640_BadPixelsCorrection((&mlx90640)->outlierPixels, mlx90640To, 1, &mlx90640);

//         printf("Subpage: %d\n", subpage);
//         //MLX90640_SetSubPage(MLX_I2C_ADDR,!subpage);

//         static float resized[OUTPUT_W * OUTPUT_H];
//         interpolate_image(mlx90640To, 24, 32, resized, OUTPUT_W, OUTPUT_H);




        
//         //display
//         for(int x = 0; x < 32; x++){
//             for(int y = 0; y < 24; y++){
//                 //std::cout << image[32 * y + x] << ",";
//                 float val = mlx90640To[32 * (23-y) + x];
//                 if(val > 99.99) val = 99.99;
//                 if(val > 53.0){
//                     printf(ANSI_COLOR_RED FMT_STRING ANSI_COLOR_RESET, val);
//                 }
//                 else if (val > 48.0){
//                     printf(ANSI_COLOR_ORANGE FMT_STRING ANSI_COLOR_YELLOW, val);
//                 }
//                 else if (val > 42.0){
//                     printf(ANSI_COLOR_YELLOW FMT_STRING ANSI_COLOR_YELLOW, val);
//                 }
//                 else if (val > 36.0) {
//                     printf(ANSI_COLOR_CYAN FMT_STRING ANSI_COLOR_RESET, val);
//                 }
//                 else {
//                     printf(ANSI_COLOR_BLUE FMT_STRING ANSI_COLOR_RESET, val);
//                 }
//             }
//             std::cout << std::endl;
//         }
//         //std::this_thread::sleep_for(std::chrono::milliseconds(20));
//         printf("\x1b[33A");
//     }
//     return 0;
// }
